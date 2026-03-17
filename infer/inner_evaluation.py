import logging
import os
import sys
import csv  # 保留csv库用于保存结果
import multiprocessing as mp
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import cv2
from infer.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from .util import metric, normal_utils
from .util.alignment import (align_depth_least_square, depth2disparity, disparity2depth, depth2log_space, log_space2depth)
from .util.metric import MetricTracker
from infer.image_utils import colorize_depth_map

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
]


def save_visualization_worker(save_vis_path, safe_pred_name, cfg_suffix, depth_pred_np, depth_raw_np, valid_mask_np, input_rgb_data, rank):
    """
    可视化保存的工作函数，在独立进程中运行
    Args:
        save_vis_path: 保存路径
        safe_pred_name: 安全的预测文件名
        cfg_suffix: cfg后缀
        depth_pred_np: 预测深度图 numpy数组
        depth_raw_np: GT深度图 numpy数组
        valid_mask_np: 有效掩码 numpy数组
        input_rgb_data: 输入RGB图像数据
        rank: GPU rank
    """
    try:
        # 转换为torch tensor用于colorize_depth_map
        depth_pred_ts = torch.from_numpy(depth_pred_np)
        depth_raw_ts = torch.from_numpy(depth_raw_np)
        valid_mask_ts = torch.from_numpy(valid_mask_np)
        
        # 1. 保存预测深度图
        depth_pred_vis = colorize_depth_map(depth_pred_ts)
        pred_save_path = os.path.join(save_vis_path, f"{safe_pred_name}{cfg_suffix}_pred.png")
        depth_pred_vis.save(pred_save_path)
        print(f"saved: {pred_save_path}")
        # 3. 保存误差图
        # 计算绝对相对误差
        abs_rel_error = torch.abs(depth_pred_ts - depth_raw_ts) / (depth_raw_ts + 1e-6)
        abs_rel_error = abs_rel_error * valid_mask_ts.float()
        
        # 使用matplotlib生成误差图
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        error_np = abs_rel_error.numpy()
        # 设置误差显示范围
        vmax = 0.2  # 可以根据需要调整
        error_normalized = np.clip(error_np / vmax, 0, 1)
        
        # 应用颜色映射
        jet_cmap = cm.get_cmap('jet')
        error_colored = jet_cmap(error_normalized)[:, :, :3]  # 去掉alpha通道
        error_colored = (error_colored * 255).astype(np.uint8)
        
        # 将无效区域设为黑色
        error_colored[~valid_mask_np] = [0, 0, 0]
        
        error_save_path = os.path.join(save_vis_path, f"{safe_pred_name}{cfg_suffix}_error.png")
        plt.imsave(error_save_path, error_colored)
        print(f"saved: {error_save_path}")
        # 关闭matplotlib figure以释放内存
        plt.close('all')
        
    except Exception as e:
        print(f"[VIS-Worker-{rank}] 可视化保存失败: {e}", file=sys.stderr)


def prepare_input_rgb_data(input_rgb):
    """
    预处理输入RGB数据，转换为numpy格式供子进程使用
    """
    if input_rgb is None:
        return None
        
    try:
        # 处理不同格式的输入图像
        if isinstance(input_rgb, torch.Tensor):
            # 如果是torch tensor，转换为numpy
            if input_rgb.dim() == 4:  # Batch dimension
                input_rgb = input_rgb[0]
            if input_rgb.dim() == 3 and input_rgb.shape[0] == 3:  # CHW格式
                input_rgb = input_rgb.permute(1, 2, 0)
            
            # 确保值在[0,1]范围内
            if input_rgb.max() <= 1.0:
                input_rgb = (input_rgb * 255).clamp(0, 255).byte()
            
            input_rgb_np = input_rgb.cpu().numpy()
            
        elif isinstance(input_rgb, np.ndarray):
            input_rgb_np = input_rgb
            # 确保值在正确范围内
            if input_rgb_np.max() <= 1.0:
                input_rgb_np = (input_rgb_np * 255).astype(np.uint8)
        else:
            # 假设是PIL图像或其他格式
            input_rgb_np = np.array(input_rgb)
            
        return input_rgb_np.copy()  # 创建副本避免进程间共享问题
        
    except Exception as e:
        print(f"处理输入RGB数据失败: {e}", file=sys.stderr)
        return None


def evaluate_single_prediction(pred_depth, depth_raw, valid_mask, dataset, device, metric_funcs, alignment_max_res=None, save_pred_vis=False, save_vis_path=None, pred_name=None, cfg_suffix="", alignment="least_square", rank=0, input_rgb=None):
    """Args:  pred_depth: 预测的深度图 (numpy array, [0,1])        depth_raw: 真实深度图 (numpy array)         valid_mask: 有效掩码 (numpy array)        dataset: 数据集对象        device: 计算设备        metric_funcs: 评估指标函数列表        alignment_max_res: 对齐时的最大分辨率        save_pred_vis: 是否保存可视化结果        save_vis_path: 可视化保存路径        pred_name: 预测文件名        cfg_suffix: cfg后缀，用于区分不同的cfg设置
    alignment: 对齐方式，可选值为"least_square"或"least_square_disparity"
    rank: GPU rank，用于多进程图像保存
    input_rgb: 输入RGB图像 (torch.Tensor or PIL.Image or numpy.ndarray)
    Returns: sample_metric: 该样本的所有评估指标列表"""
    # 确保预测深度图的维度正确
    if len(pred_depth.shape) == 3:
        pred_depth = pred_depth.mean(0)  # [0,1]

    # 调整预测深度图尺寸以匹配真实深度图
    if pred_depth.shape != depth_raw.shape:
        pred_depth = cv2.resize(pred_depth, (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=pred_depth,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
    elif "log_space" == alignment:
        gt_log, gt_non_neg_mask = depth2log_space(depth=depth_raw, return_mask=True)
        pred_non_neg_mask = pred_depth > 0
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        # 确保输入是numpy数组类型
        if isinstance(gt_log, torch.Tensor):
            gt_log = gt_log.cpu().numpy()

        log_space_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_log,
            pred_arr=pred_depth,
            valid_mask_arr=valid_nonnegative_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
        log_space_pred = np.clip(log_space_pred, a_min=None, a_max=5.)
        depth_pred = log_space2depth(log_space_pred)
    # 裁剪到数据集的深度范围
    depth_pred = np.clip(depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth)

    # 裁剪到 d > 0 以便评估
    depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

    # 转换到设备进行评估
    depth_pred_ts = torch.from_numpy(depth_pred).to(device)
    depth_raw_ts = torch.from_numpy(depth_raw).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)

    # 启动可视化保存进程（同步）
    if save_pred_vis and save_vis_path is not None and pred_name is not None:
        safe_pred_name = pred_name.replace('/', '_').replace('\\', '_')
        input_rgb_data = prepare_input_rgb_data(input_rgb)
        vis_process = mp.Process(
            target=save_visualization_worker,
            args=(
                save_vis_path,
                safe_pred_name,
                cfg_suffix,
                depth_pred.copy(),
                depth_raw.copy(),
                valid_mask.copy(),
                input_rgb_data,
                rank
            )
        )
        vis_process.start()
        # save_visualization_worker(save_vis_path, safe_pred_name, cfg_suffix, depth_pred.copy(), depth_raw.copy(), valid_mask.copy(), input_rgb_data, rank)

    # 计算评估指标
    sample_metric = []
    for met_func in metric_funcs:
        _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
        sample_metric.append(_metric)

    return sample_metric


def evaluation_depth_custom_parallel(rank, world_size, output_dir, dataset_config, args, pipeline, base_data_dir, pred_suffix="", alignment="least_square", alignment_max_res=None, prediction_dir=None, save_pred_vis=False):
    """
    支持多GPU并行的深度评估函数
    """
    import time

    os.makedirs(output_dir, exist_ok=True)

    cuda_avail = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}")

    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL, prompt_type=args.prompt_type)

    # 获取数据集名称，用于CSV表命名
    dataset_name = dataset.__class__.__name__

    # 初始化存储结果的数据列表
    results_data = []

    # 计算每个GPU处理的数据范围
    total_samples = len(dataset)
    if args.num_samples > 0:
        total_samples = min(args.num_samples, total_samples)

    chunk_size = total_samples // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else total_samples

    from torch.utils.data import SubsetRandomSampler
    indices = list(range(start_idx, end_idx))

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0 if args.debug else 4, pin_memory=True, sampler=SubsetRandomSampler(indices),shuffle=False)

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    # 为cfg=1和cfg=6分别创建metric tracker
    metric_tracker_Lpred = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker_Lpred.reset()
    metric_tracker_Rpred = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker_Rpred.reset()

    if save_pred_vis:
        save_vis_path = os.path.join(output_dir, "vis")
        os.makedirs(save_vis_path, exist_ok=True)

        # 创建CSV保存目录 - 每个卡都创建
        csv_save_path = os.path.join(output_dir, "csv_results")
        os.makedirs(csv_save_path, exist_ok=True)
    else:
        save_vis_path = None
        csv_save_path = None

    processing_times = []
    vis_processes = []  # 用于管理可视化进程
    max_vis_processes = 4  # 限制同时运行的可视化进程数量

    sample_count = 0
    for data in dataloader:
        sample_count += 1

        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        # Get predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=pred_suffix)
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)

        start_time = time.time()
        image_list, Lpred, Rpred = pipeline.generate_image(args.prompt if args.prompt_type == "query" else data["prompt"][0], negative_prompt="", ref_images=data["rgb"], num_samples=1, num_steps=args.num_steps, cfg_guidance=args.cfg_guidance, seed=args.seed + rank, show_progress=False, size_level=args.size_level, args=args)
        end_time = time.time()
        processing_times.append(end_time - start_time)

        Lpred = Lpred[0].cpu().numpy()
        
        # 保存可视化结果（使用新的多进程方式）
        if save_pred_vis and save_vis_path is not None:
            # 清理文件名，替换路径分隔符
            safe_pred_name = pred_name.replace('/', '_').replace('\\', '_')
            
            # 预处理输入RGB数据
            input_rgb_data = prepare_input_rgb_data(data["rgb"])
            
            # 限制同时运行的可视化进程数量
            while len([p for p in vis_processes if p.is_alive()]) >= max_vis_processes:
                # 等待一些进程完成
                for p in vis_processes[:]:
                    if not p.is_alive():
                        vis_processes.remove(p)
                if len([p for p in vis_processes if p.is_alive()]) >= max_vis_processes:
                    time.sleep(0.1)  # 短暂等待
            
            # 创建子进程进行可视化保存
            vis_process = mp.Process(
                target=save_visualization_worker,
                args=(
                    save_vis_path,
                    safe_pred_name,
                    "_Lpred",
                    Lpred.copy(),
                    depth_raw.copy(),
                    valid_mask.copy(),
                    input_rgb_data,
                    rank
                )
            )
            vis_process.start()
            vis_processes.append(vis_process)
        
        sample_metric_Lpred = evaluate_single_prediction(pred_depth=Lpred, depth_raw=depth_raw, valid_mask=valid_mask, dataset=dataset, device=device, metric_funcs=metric_funcs, alignment_max_res=alignment_max_res, save_pred_vis=False, save_vis_path=None, pred_name=pred_name, cfg_suffix="_Lpred", alignment=alignment, rank=rank, input_rgb=data["rgb"])

        for i, met_func in enumerate(metric_funcs):
            metric_name = met_func.__name__
            metric_tracker_Lpred.update(metric_name, sample_metric_Lpred[i])

        # 输出每个样本的结果
        img_id = os.path.basename(rgb_name).replace('.png', '').replace('.jpg', '')
        global_sample_idx = start_idx + sample_count

        # CFG=1结果
        abs_rel_Lpred = sample_metric_Lpred[0]  # abs_relative_difference
        rmse_Lpred = sample_metric_Lpred[2]  # rmse_linear
        delta1_Lpred = sample_metric_Lpred[4]  # delta1_acc

        # 修改输出格式
        if args.save_viz:
            print(f"|{global_sample_idx:03d}|{abs_rel_Lpred:.4f}|{rmse_Lpred:.4f}|{delta1_Lpred:.4f}|", file=sys.stderr)
        elif not args.save_viz:
            print(f"[GPU:{rank}] 样本:{global_sample_idx:03d}/{total_samples} | ID:{img_id:<12}", file=sys.stderr)
            print(f"  CFG=1: abs_rel:{abs_rel_Lpred:.4f} | rmse:{rmse_Lpred:.4f} | a1:{delta1_Lpred:.4f}", file=sys.stderr)
            print(f"  时间: {processing_times[-1]:.2f}s", file=sys.stderr)

        # 所有卡都保存结果到列表
        if args.save_viz:
            results_data.append({'GPU_Rank': rank, 'Sample_ID': global_sample_idx, 'Image_Name': rgb_name, 'abs_rel': abs_rel_Lpred, 'rmse': rmse_Lpred, 'delta1': delta1_Lpred, 'processing_time': processing_times[-1]})

    # 等待所有可视化进程完成
    if save_pred_vis:
        print(f"[GPU:{rank}] 等待可视化进程完成...", file=sys.stderr)
        for p in vis_processes:
            p.join(timeout=30)  # 设置超时时间
            if p.is_alive():
                print(f"[GPU:{rank}] 可视化进程超时，强制终止", file=sys.stderr)
                p.terminate()
        print(f"[GPU:{rank}] 所有可视化进程已完成", file=sys.stderr)

    if args.save_viz and csv_save_path is not None:
        csv_file_path = os.path.join(csv_save_path, f"{dataset_name}_results_rank{rank}.csv")

        try:
            with open(csv_file_path, 'w', newline='') as csvfile:
                fieldnames = ['GPU_Rank', 'Sample_ID', 'Image_Name', 'abs_rel', 'rmse', 'delta1', 'processing_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_data:
                    writer.writerow(row)
            print(f"[GPU:{rank}] 结果已保存至CSV: {csv_file_path}", file=sys.stderr)
        except Exception as e:
            print(f"[GPU:{rank}] 保存CSV失败: {e}", file=sys.stderr)

    return metric_tracker_Lpred, metric_tracker_Rpred, processing_times


def evaluation_normal_custom_parallel(rank, world_size, output_dir, base_data_dir, dataset_split_path, pipeline, args, eval_datasets, save_pred_vis=False):
    """
    支持多GPU并行的normal评估函数
    """
    import time

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{rank}")

    # 为每个数据集创建结果字典
    all_normal_errors = {}
    all_processing_times = {}
    all_dataset_metrics = {}

    for dataset_name, split in eval_datasets:
        # 创建数据加载器 - 减少num_workers避免资源竞争
        try:
            # 创建数据集
            from infer.dataset_normal.normal_dataloader import NormalDataset
            dataset = NormalDataset(base_data_dir, dataset_split_path, dataset_name=dataset_name, split=split, mode='test', epoch=0)

            total_samples = len(dataset)
            if args.num_samples > 0:
                total_samples = min(args.num_samples, total_samples)

            # 计算当前GPU需要处理的样本范围
            samples_per_gpu = total_samples // world_size
            start_idx = rank * samples_per_gpu
            if rank == world_size - 1:
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_gpu

            # 创建样本索引并使用SubsetRandomSampler
            from torch.utils.data import SubsetRandomSampler
            indices = list(range(start_idx, end_idx))

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, sampler=SubsetRandomSampler(indices))

            if rank == 0:
                print(f"[GPU:{rank}] 开始评估Normal数据集: {dataset_name}")

        except Exception as e:
            print(f"[GPU:{rank}] 创建数据加载器失败: {e}")
            continue

        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        if save_pred_vis:
            save_vis_path = os.path.join(dataset_output_dir, "vis")
            os.makedirs(save_vis_path, exist_ok=True)
        else:
            save_vis_path = None

        processing_times = []
        total_normal_errors = None
        sample_count = 0
        vis_processes = []  # 用于管理normal可视化进程
        max_vis_processes = 5  # 限制同时运行的可视化进程数量（normal可视化更耗内存）

        for data_dict in dataloader:
            sample_count += 1

            img = data_dict['img'].to(device)
            scene_names = data_dict['scene_name']
            img_names = data_dict['img_name']

            # 获取原始图像尺寸
            _, _, orig_H, orig_W = img.shape

            start_time = time.time()
            image_list,L_pred , norm_out = pipeline.generate_image("Predict the depth map for the image on the left and the normal map on the right.", negative_prompt="", ref_images=img, num_samples=1, num_steps=args.num_steps, cfg_guidance=args.cfg_guidance, seed=args.seed + rank, show_progress=False, size_level=args.size_level, args=args, judge=data_dict['normal'].to(device) if dataset_name == "vkitti" or dataset_name == "hypersim" else None, name=img_names)
            end_time = time.time()
            processing_times.append(end_time - start_time)

            # 处理normal输出
            norm_out = torch.nn.functional.interpolate(norm_out, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            norm = torch.linalg.norm(norm_out, axis=1, keepdims=True)
            norm[norm < 1e-9] = 1e-9
            norm_out = norm_out / norm

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            # 计算误差（如果有ground truth）
            # if 'normal' in data_dict.keys():
            gt_norm = data_dict['normal'].to(device)
            gt_norm_mask = data_dict['normal_mask'].to(device)

            pred_error = normal_utils.compute_normal_error(pred_norm, gt_norm)
            if total_normal_errors is None:
                total_normal_errors = pred_error[gt_norm_mask]
            else:
                total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            # 保存可视化结果（使用新的多进程方式）
            if save_vis_path is not None:
                # 限制同时运行的可视化进程数量
                while len([p for p in vis_processes if p.is_alive()]) >= max_vis_processes:
                    # 等待一些进程完成
                    for p in vis_processes[:]:
                        if not p.is_alive():
                            vis_processes.remove(p)
                    if len([p for p in vis_processes if p.is_alive()]) >= max_vis_processes:
                        time.sleep(0.1)  # 短暂等待
                
                prefixs = ['%s_%s' % (i, j) for (i, j) in zip(scene_names, img_names)]
                
                # 预处理数据
                img_data, pred_norm_data, pred_kappa_data, gt_norm_data, gt_norm_mask_data, pred_error_data = prepare_normal_data_for_process(
                    img, pred_norm, pred_kappa, gt_norm, gt_norm_mask, pred_error
                )
                
                if img_data is not None:  # 确保数据预处理成功
                    # 创建子进程进行可视化保存
                    vis_process = mp.Process(
                        target=save_normal_visualization_worker,
                        args=(
                            save_vis_path,
                            prefixs,
                            img_data,
                            pred_norm_data,
                            pred_kappa_data,
                            gt_norm_data,
                            gt_norm_mask_data,
                            pred_error_data,
                            rank
                        )
                    )
                    vis_process.start()
                    vis_processes.append(vis_process)

            # 输出进度信息
            global_sample_idx = start_idx + sample_count
            img_id = '_'.join([scene_names[0], img_names[0]])

            if rank == 0 or sample_count % 10 == 0:  # 减少输出频率
                print(f"[GPU:{rank}] | 样本:{global_sample_idx:03d} | ID:{img_id} | 时间:{processing_times[-1]:.2f}s| ", file=sys.stderr)

        # 等待所有可视化进程结束
        if save_pred_vis:
            print(f"[GPU:{rank}] 等待Normal可视化进程完成...", file=sys.stderr)
            for p in vis_processes:
                p.join(timeout=60)  # normal可视化需要更长时间，设置60秒超时
                if p.is_alive():
                    print(f"[GPU:{rank}] Normal可视化进程超时，强制终止", file=sys.stderr)
                    p.terminate()
            print(f"[GPU:{rank}] 所有Normal可视化进程已完成", file=sys.stderr)

        # 计算当前GPU的指标
        metrics = None
        if total_normal_errors is not None and len(total_normal_errors) > 0:
            metrics = normal_utils.compute_normal_metrics(total_normal_errors)
            if rank == 0:
                print(f"[GPU:{rank}] 数据集 {dataset_name} 部分结果:")
                print("mean median rmse 5 7.5 11.25 22.5 30")
                print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (metrics['mean'], metrics['median'], metrics['rmse'], metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

        # 存储结果
        all_normal_errors[dataset_name] = total_normal_errors.cpu() if total_normal_errors is not None else None
        all_processing_times[dataset_name] = processing_times
        all_dataset_metrics[dataset_name] = metrics

    return all_normal_errors, all_processing_times, all_dataset_metrics

def save_normal_visualization_worker(save_vis_path, prefixs, img_data, pred_norm_data, pred_kappa_data, gt_norm_data, gt_norm_mask_data, pred_error_data, rank):
    """
    Normal可视化保存的工作函数，在独立进程中运行
    Args:
        save_vis_path: 保存路径
        prefixs: 文件名前缀列表
        img_data: 输入图像数据 (numpy array)
        pred_norm_data: 预测normal数据 (numpy array)
        pred_kappa_data: 预测kappa数据 (numpy array or None)
        gt_norm_data: GT normal数据 (numpy array)
        gt_norm_mask_data: GT normal掩码数据 (numpy array)
        pred_error_data: 预测误差数据 (numpy array)
        rank: GPU rank
    """
    try:
        import infer.visualize as vis_utils

        # 转换为torch tensor用于可视化函数
        img_ts = torch.from_numpy(img_data)
        pred_norm_ts = torch.from_numpy(pred_norm_data)
        pred_kappa_ts = torch.from_numpy(pred_kappa_data) if pred_kappa_data is not None else None
        gt_norm_ts = torch.from_numpy(gt_norm_data)
        gt_norm_mask_ts = torch.from_numpy(gt_norm_mask_data)
        pred_error_ts = torch.from_numpy(pred_error_data)
        
        # 使用matplotlib的非交互式后端
        import matplotlib
        matplotlib.use('Agg')
        
        # 调用可视化函数
        vis_utils.visualize_normal(save_vis_path, prefixs, img_ts, pred_norm_ts, pred_kappa_ts, gt_norm_ts, gt_norm_mask_ts, pred_error_ts)
        
    except Exception as e:
        print(f"[NORMAL-VIS-Worker-{rank}] Normal可视化保存失败: {e}", file=sys.stderr)


def prepare_normal_data_for_process(img, pred_norm, pred_kappa, gt_norm, gt_norm_mask, pred_error):
    """
    预处理normal数据，转换为numpy格式供子进程使用
    """
    try:
        img_data = img.cpu().numpy()
        pred_norm_data = pred_norm.cpu().numpy()
        pred_kappa_data = pred_kappa.cpu().numpy() if pred_kappa is not None else None
        gt_norm_data = gt_norm.cpu().numpy()
        gt_norm_mask_data = gt_norm_mask.cpu().numpy()
        pred_error_data = pred_error.cpu().numpy()
        
        return img_data, pred_norm_data, pred_kappa_data, gt_norm_data, gt_norm_mask_data, pred_error_data
    except Exception as e:
        print(f"处理Normal数据失败: {e}", file=sys.stderr)
        return None, None, None, None, None, None
