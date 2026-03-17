import argparse
import json
import itertools
import math
import os
import sys
import time
from pathlib import Path

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm
import torch.nn.functional as Func

import infer.sampling as sampling
from modules.autoencoder import AutoEncoder
from modules.model_edit import Step1XParams, Step1XEdit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_DIR = REPO_ROOT / "Qwen"
EMPTY_PROMPT_LATENT_PATH = REPO_ROOT / "latent" / "no_info.npz"


def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=assign)
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model


def load_models(dit_path=None, ae_path=None, qwen2vl_model_path=None, device="cuda", max_length=256, dtype=torch.bfloat16, args=None):
    empty_llm = args is not None and hasattr(args, 'prompt_type') and args.prompt_type == 'empty'
    if empty_llm:
        print("[INFO] prompt_type=empty, 跳过Qwen模型加载")
        qwen2vl_encoder = None
    else:
        # Lazy import to avoid pulling transformers/vision stack during evaluation with prompt_type=empty.
        from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
        qwen2vl_encoder = Qwen2VLEmbedder(
            qwen2vl_model_path,
            device=device,
            max_length=max_length,
            dtype=dtype,
            args=args,
        )

    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path, 'cpu')
    dit = load_state_dict(dit, dit_path, 'cpu')

    ae = ae.to(dtype=torch.float32)

    return ae, dit, qwen2vl_encoder


def equip_dit_with_lora_sd_scripts(ae, text_encoders, dit, lora, device='cuda'):
    from safetensors.torch import load_file
    weights_sd = load_file(lora)
    is_lora = True
    from library import lora_module
    module = lora_module
    lora_model, _ = module.create_network_from_weights(1.0, None, ae, text_encoders, dit, weights_sd, True)
    lora_model.merge_to(text_encoders, dit, weights_sd)

    lora_model.set_multiplier(1.0)
    return lora_model

class ImageGenerator:

    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
        quantized=False,
        offload=False,
        lora=None,
        args=None,
    ) -> None:
        self.device = torch.device(device)
        self.args = args
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
            device=device,
            args=args,
        )
        if not quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(dtype=torch.float8_e4m3fn)
        if not offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)
        self.quantized = quantized
        self.offload = offload
        if lora is not None:
            self.lora_module = equip_dit_with_lora_sd_scripts(
                self.ae,
                [self.llm_encoder],
                self.dit,
                lora,
                device=self.dit.device,
            )
        else:
            self.lora_module = None

    def prepare(self, prompt, img, ref_image, ref_image_raw, empty_llm=False):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)  #2,16,82,110->2,2255,64
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)  # 将二维图像"压平"成一维序列 这是为 Transformer 模型准备的，因为它处理的是序列数据
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        #img 和 ref_img 已经不再是二维的图片了，而是变成了一个 "patches" (图像块) 的序列。一个块是64维度的。Transformer不知道这2255个图像块哪个在左上角，哪个在右下角。
        img_ids = torch.zeros(h // 2, w // 2, 3)  #41,55,3 # h 和 w 是潜在空间的高和宽，但 rearrange 操作把 2x2 的小块合并了# 所以实际的网格大小是 h/2 x w/2# 最后的 3 代表每个坐标有3个分量 (一个预留, Y坐标, X坐标)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]  #通过广播机制，第 i 行的所有点的第二个分量都被赋值为 i
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)  #将二维坐标网格"压平"成一维序列，并复制到对应的batch size

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)

        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]

        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)

        if empty_llm:
            empty_prompt_cache = getattr(self.args, "empty_prompt_cache", None) if self.args is not None else None
            cache_path = Path(empty_prompt_cache) if empty_prompt_cache else EMPTY_PROMPT_LATENT_PATH
            data = np.load(cache_path)
            txt = torch.from_numpy(data['embeds']).to(img.device).unsqueeze(0)
            txt = torch.cat([txt, txt], dim=0)
            mask = torch.from_numpy(data['masks']).to(img.device).unsqueeze(0)
            mask = torch.cat([mask, mask], dim=0)
        else:
            txt, mask = self.llm_encoder(prompt, ref_image_raw)  #之所以都要复制一份，是因为有正负两种prompt

        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)  #2,4550,64 在patch上concat？？？
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),  #图像坐标
            "llm_embedding": txt.to(img.device),  #文字向量
            "txt_ids": txt_ids.to(img.device),  #文字坐标
        }

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 6.0,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='denoising...')
        else:
            pbar = itertools.pairwise(timesteps)
        '''
        Cond 0 RGB
        Uncd 0 RGB
        '''
        for t_curr, t_prev in pbar:
            '''
            若输入维度是2，无所谓，维度是1则：
            imgN D RGB
            imgN D RGB
            '''
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            pred, feat = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )

            assert cfg_guidance != -1, " cfg_guidance must not be -1 NOW!!!!"
            cond, uncond = (
                pred[0:pred.shape[0] // 2, :],
                pred[pred.shape[0] // 2:, :],
            )
            '''
            Cond D ??? <- pred
            Uncd D ???
            '''
            pred = uncond + cfg_guidance * (cond - uncond)  #1,4608,64
            pred1 = cond  #todo only support single denoise!!!
            '''
                Cond 0 RGB
              + pred D ???
                temI D ??? 
            '''
            tem_img = img[0:img.shape[0] // 2, :] + (t_prev - t_curr) * pred  #1,4608,64
            img_input_length = img.shape[1] // 2
            '''
                tmpI [D](√)  ???(x)
            cat Cond  0(x)   [RGB](√)
                imgN  [D]    [RGB]
            '''
            img = torch.cat(
                [
                    tem_img[:, :img_input_length],  #1,2304,64
                    img[:img.shape[0] // 2, img_input_length:],  #1,2304,64
                ],
                dim=1)  #1,4608,64
        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return img[:, :img.shape[1] // 2], pred1[:, img.shape[1] // 2:]

    def double_denoise(self,img,img_ids,llm_embedding,txt_ids,timesteps,cfg_guidance=6.0,mask=None,height=None,width=None):
        if img.shape[0] == 1 and cfg_guidance != -1:
            img = torch.cat([img, img], dim=0)
        
        t_vec = torch.full((img.shape[0],), 1.0, dtype=img.dtype, device=img.device)

        pred, _ = self.dit(
            img=img,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=t_vec,
            llm_embedding=llm_embedding,
            t_vec=t_vec,
            mask=mask,
        )

        assert cfg_guidance != -1, " cfg_guidance must not be -1 NOW!!!!"
        pred, uncond = (
            pred[0:pred.shape[0] // 2, :],
            pred[pred.shape[0] // 2:, :],
        )
        Lpred,Rpred = self.unpack_latents(pred, height//16, width//16)
        return Lpred.to(torch.float32),Rpred.to(torch.float32)

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )
        
    @staticmethod
    def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int):
            """
            x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
            """
            import einops
            x = einops.rearrange(x, "b (p h w) (c ph pw) -> b p c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2, p=2)
            return x[:, 0], x[:, 1]
        
    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image

    def input_process_image(self, img):
        if isinstance(img, torch.Tensor):
            w, h = img.shape[-1], img.shape[-2]
        elif isinstance(img, Image.Image):
            w, h = img.size

        if w <= 1024 and h <= 768:
            w_new, h_new = 1024, 768
        elif w <= 1280 and h <= 960:
            w_new, h_new = 1216, 352
        elif w <= 6048 and h <= 4032:
            w_new, h_new = 864, 576
        else:
            w_new, h_new = w, h

        if isinstance(img, torch.Tensor):
            img_resized = Func.interpolate(img, (h_new, w_new), mode='bilinear', align_corners=False)
            img_resized = img_resized.clamp(0, 1)
        else:
            img_resized = img.resize((w_new, h_new))

        return img_resized, (w_new, h_new)

    @torch.inference_mode()
    def generate_image(
        self,prompt,negative_prompt,ref_images,num_steps,cfg_guidance,seed,num_samples=1,init_image=None,image2image_strength=0.0,show_progress=False,size_level=512,args=None,judge=None,name=None
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."

        ref_images_raw, img_info = self.input_process_image(ref_images)
        if isinstance(ref_images, Image.Image):
            ref_images_raw = self.load_image(ref_images_raw)

        height, width = ref_images_raw.shape[-2], ref_images_raw.shape[-1]

        ref_images_raw = ref_images_raw.to(self.device)
        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)  #bs,3,656,880 -> 1,16,82,110
        #加入cache
        
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        t0 = time.perf_counter()

        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae = self.ae.to(self.device)
            init_image = self.ae.encode(init_image.to() * 2 - 1)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

        if args is not None and hasattr(args, 'single_denoise') and not args.single_denoise:
            x = torch.randn(num_samples,16,height // 8,width // 8,device=self.device,dtype=torch.bfloat16,generator=torch.Generator(device=self.device).manual_seed(seed),)
        else:
            x= torch.zeros(num_samples,16,height // 8,width // 8,device=self.device,dtype=torch.bfloat16,)


        timesteps = sampling.get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        x = torch.cat([x, x], dim=0)
        ref_images = torch.cat([ref_images, ref_images], dim=0)  #这里是为了有无prompt
        ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
        
        # 检查args和prompt_type属性
        empty_llm = args is not None and hasattr(args, 'prompt_type') and args.prompt_type == 'empty'
        
        inputs = self.prepare(
            [prompt, negative_prompt],
            x,  #img这个gt给的是全噪声在推理
            ref_image=ref_images,
            ref_image_raw=ref_images_raw,
            empty_llm=empty_llm)

        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16):
            # Lpred,Rpred = self.double_denoise(**inputs,cfg_guidance=cfg_guidance,timesteps=timesteps,height=height,width=width)#图像中包括ref image
            Lpred,Rpred = self.denoise(**inputs,cfg_guidance=cfg_guidance,timesteps=timesteps,show_progress=show_progress,timesteps_truncate=1.0,)#图像中包括ref image
            Lpred=self.unpack(Lpred.float(),height,width)
            Rpred=self.unpack(Rpred.float(),height,width)
        if judge is not None:
            judge = Func.interpolate(judge, (height, width), mode='bilinear', align_corners=False)
            training_gt=self.ae.encode(judge)
            traing_loss = torch.nn.functional.mse_loss(Rpred,training_gt)
            print(f"training_loss with rgb2: {traing_loss}")
            
            norm = torch.linalg.norm(judge, dim=1, keepdim=True)
            norm[norm < 1e-9] = 1e-9
            judge = judge / norm
            training_gt =self.ae.encode(judge)
            training_loss = torch.nn.functional.mse_loss(Rpred,training_gt)
            print(f"training_loss with normed_rgb: {training_loss}")
        Lpred = self.ae.decode(Lpred)
        Rpred = self.ae.decode(Rpred)
        
            
        Lpred = Lpred.clamp(-1, 1)
        Lpred = Lpred.mul(0.5).add(0.5)
        Rpred = Rpred.clamp(-1, 1)
        # Rpred = Rpred.mul(0.5).add(0.5)

        images_list = []
        for img in Rpred.float():
            images_list.append(self.output_process_image(F.to_pil_image(img), img_info))
        return images_list, Lpred.float(), Rpred.float()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output image directory')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing image names and prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=512, type=int)
    parser.add_argument('--offload', action='store_true', help='Use offload for large models')
    parser.add_argument('--quantized', action='store_true', help='Use fp8 model weights')
    parser.add_argument('--lora', type=str, default=None)
    parser.add_argument('--qwen2vl_model_path', type=str, default=str(DEFAULT_QWEN_DIR), help='Path to the local Qwen2.5-VL model directory')
    parser.add_argument('--empty_prompt_cache', type=str, default=str(EMPTY_PROMPT_LATENT_PATH), help='Path to the empty-prompt latent cache')
    args = parser.parse_args()

    assert os.path.exists(args.input_dir), f"Input directory {args.input_dir} does not exist."
    assert os.path.exists(args.json_path), f"JSON file {args.json_path} does not exist."

    args.output_dir = args.output_dir.rstrip('/') + ('-offload' if args.offload else "") + ('-quantized' if args.quantized else "") + f"-{args.size_level}"
    os.makedirs(args.output_dir, exist_ok=True)

    image_and_prompts = json.load(open(args.json_path, 'r'))

    image_edit = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258-FP8.safetensors" if args.quantized else "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=args.qwen2vl_model_path,
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        lora=args.lora,
    )

    time_list = []
    for image_name, prompt in image_and_prompts.items():
        image_path = os.path.join(args.input_dir, image_name)
        output_path = os.path.join(args.output_dir, image_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        start_time = time.time()

        images, _, _ = image_edit.generate_image(
            prompt,
            negative_prompt="",
            ref_images=Image.open(image_path).convert("RGB"),
            num_samples=1,
            num_steps=args.num_steps,
            cfg_guidance=args.cfg_guidance,
            seed=args.seed,
            show_progress=True,
            size_level=args.size_level,
        )

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        time_list.append(time.time() - start_time)

        images[0].save(output_path, lossless=True)
    if len(time_list) > 1:
        print(f'average time for {args.output_dir}: ', sum(time_list[1:]) / len(time_list[1:]))


if __name__ == "__main__":
    main()
