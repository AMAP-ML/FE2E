import argparse
import os
import re
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from infer.seed_all import seed_all

# 设置环境变量消除tokenizers警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_DEBUG'] = 'WARN'
# 消除torchvision警告
os.environ['TORCHVISION_DISABLE_DEPRECATED_WARNING'] = '1'

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QWEN_DIR = os.path.join(REPO_ROOT, "Qwen")
DEFAULT_DEPTH_DATASET_CONFIGS = {
    "nyu_v2": "configs/data_nyu_test.yaml",
    "kitti": "configs/data_kitti_eigen_test.yaml",
    "eth3d": "configs/data_eth3d.yaml",
    "diode": "configs/data_diode_all.yaml",
    "scannet": "configs/data_scannet_val.yaml",
}
DEFAULT_NORMAL_DATASETS = {
    "nyuv2": "test",
    "scannet": "test",
    "ibims": "ibims",
    "sintel": "sintel",
    "oasis": "val",
    "hypersim": "hypersim",
}


def resolve_eval_data_root(args, *required_markers):
    """Resolve the evaluation data root without depending on the launch cwd."""
    candidates = []
    if getattr(args, "eval_data_root", None):
        candidates.append(os.path.abspath(args.eval_data_root))

    candidates.extend(
        [
            os.path.join(REPO_ROOT, "infer"),
            os.path.join(REPO_ROOT, "data"),
            os.path.join(os.path.dirname(REPO_ROOT), "data"),
        ]
    )

    for candidate in candidates:
        if all(os.path.exists(os.path.join(candidate, marker)) for marker in required_markers):
            return candidate

    checked = ", ".join(
        os.path.join(candidate, marker) for candidate in candidates for marker in required_markers
    )
    raise FileNotFoundError(f"未找到评测数据根目录，已检查: {checked}")


def parse_depth_eval_datasets(raw_value):
    requested = [item.strip() for item in raw_value.split(",") if item.strip()]
    if requested == ["all"]:
        requested = list(DEFAULT_DEPTH_DATASET_CONFIGS.keys())
    invalid = [item for item in requested if item not in DEFAULT_DEPTH_DATASET_CONFIGS]
    if invalid:
        raise ValueError(f"不支持的 depth 数据集: {invalid}，可选值: {sorted(DEFAULT_DEPTH_DATASET_CONFIGS)}")
    return {name: DEFAULT_DEPTH_DATASET_CONFIGS[name] for name in requested}


def parse_normal_eval_datasets(raw_value):
    requested = [item.strip() for item in raw_value.split(",") if item.strip()]
    if requested == ["all"]:
        requested = list(DEFAULT_NORMAL_DATASETS.keys())
    invalid = [item for item in requested if item not in DEFAULT_NORMAL_DATASETS]
    if invalid:
        raise ValueError(f"不支持的 normal 数据集: {invalid}，可选值: {sorted(DEFAULT_NORMAL_DATASETS)}")
    return [(name, DEFAULT_NORMAL_DATASETS[name]) for name in requested]


def collect_and_merge_dual_cfg_results(rank, world_size, gathered_metrics_Lpred, gathered_times):
    """
    收集并合并双CFG配置的评估结果
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        gathered_metrics_Lpred
        gathered_times: 处理时间收集结果
    
    Returns:
        tuple: (all_metrics_Lpred, dataset_times)
    """
    if rank != 0:
        return None, None

    # 合并处理时间
    dataset_times = []
    for times_list in gathered_times:
        dataset_times.extend(times_list)

    #先处理L的
    all_metrics_L = {}
    valid_metrics_L = [m for m in gathered_metrics_Lpred if m]
    if valid_metrics_L:
        for key in valid_metrics_L[0].keys():
            values = [m[key] for m in valid_metrics_L if key in m]
            if values:
                all_metrics_L[key] = np.mean(values)

    return all_metrics_L, dataset_times


def format_dual_cfg_results_table(dataset_name, model_identifier, all_metrics_L, dataset_times):
    """
    格式化双CFG配置的结果表格
    
    Args:
        dataset_name: 数据集名称
        model_identifier: 模型标识符
        all_metrics_L: CFG=1的评估指标
        dataset_times: 处理时间列表
    
    Returns:
        str: 格式化的结果字符串
    """
    eval_metrics_order = ["abs_relative_difference", "squared_relative_difference", "rmse_linear", "rmse_log", "delta1_acc", "delta2_acc", "delta3_acc"]

    # 获取CFG=1的指标值
    mean_errors_L = [all_metrics_L.get(metric, 0.0) for metric in eval_metrics_order]

    # 构建表格
    metrics_header = ["Dataset", "Model", "CFG", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    # CFG=1的结果行
    values_data_L = [dataset_name, model_identifier, "CFG=1"] + [f"{v:.4f}" for v in mean_errors_L]

    header_line = "| " + " | ".join(metrics_header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(metrics_header)) + " |"
    values_line_L = "| " + " | ".join(values_data_L) + " |"

    # 生成输出字符串
    result_str = f"\n数据集 {dataset_name} 评估完成!\n"
    result_str += "-" * 100 + "\n"
    result_str += header_line + "\n"
    result_str += separator_line + "\n"
    result_str += values_line_L + "\n"

    # 添加统计信息
    sample_count = len(dataset_times)
    result_str += f"样本数: {sample_count}\n"
    if dataset_times:
        result_str += f"平均处理时间: {np.mean(dataset_times):.2f}秒/图像\n"
    result_str += "-" * 100 + "\n"

    return result_str


def save_dual_cfg_results_summary(output_dir, all_dataset_results, model_identifier):
    """
    保存所有数据集的双CFG评估结果摘要
    
    Args:
        output_dir: 输出目录
        all_dataset_results: 所有数据集的结果字典
    """
    summary_file = os.path.join(output_dir, f"{model_identifier}.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("双CFG配置深度评估结果汇总\n")
        f.write("=" * 120 + "\n\n")

        for dataset_name, result_data in all_dataset_results.items():
            f.write(result_data['formatted_output'])
            f.write(f"结果保存至: {result_data['eval_dir']}\n\n")

    print(f"双CFG评估结果摘要已保存至: {summary_file}")


def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(description="Run Step...")
    parser.add_argument('--model_path', type=str, default='./pretrain', help='模型路径')
    parser.add_argument('--qwen2vl_model_path', type=str, default=DEFAULT_QWEN_DIR, help='Qwen2.5-VL 模型目录')
    parser.add_argument('--eval_data_root', type=str, default=None, help='评测数据根目录，默认自动在仓库相对路径下查找')
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./infer/eval_results", help="Output directory.")
    parser.add_argument('--num_steps', type=int, default=28, help='扩散步数')
    parser.add_argument('--num_samples', type=int, default=-1, help='生成样本数')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG引导强度')
    parser.add_argument('--size_level', type=int, default=768, help='输入图像大小')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='使用的GPU数量')
    parser.add_argument('--save_viz', action='store_true', help='保存可视化结果')
    parser.add_argument('--offload', action='store_true', help='使用CPU卸载以节省GPU内存')
    parser.add_argument('--quantized', action='store_true', help='使用量化模型')
    parser.add_argument('--lora', type=str, help='LoRA模型路径')
    parser.add_argument('--single_denoise', action='store_true', default=False, help='单步推理')
    parser.add_argument('--old_prompt', action='store_true', default=False, help='使用旧版提示')
    parser.add_argument('--prompt_type', type=str, default='query', help='提示类型')
    parser.add_argument('--prompt', type=str, default='Describe the 3D structure and layout of the scene in the image. Predict the depth of this image.', help='提示')
    parser.add_argument('--norm_type', type=str, default='depth', help='预测结果的归一化方式，目前有 depth、disp、ln')
    parser.add_argument('--task_name', type=str, default='depth', help='任务名称，支持 depth 或 normal')
    parser.add_argument('--depth_eval_datasets', type=str, default='eth3d', help='逗号分隔的 depth 评测数据集')
    parser.add_argument('--normal_eval_datasets', type=str, default='nyuv2,scannet', help='逗号分隔的 normal 评测数据集')
    parser.add_argument('--debug', action='store_true', default=False, help='调试模式')
    args = parser.parse_args()
    if args.single_denoise:
        args.num_steps = 1
    return args


def extract_model_identifier(lora_path):
    """
    从lora路径中提取模型标识符
    支持多种路径格式：
    - ./log_err/dis-hvsge-log/ckpt.safetensors -> dis-hvsge-log
    - /path/to/folder/ckpt-123 -> folder-epoch123
    - /path/to/model.safetensors -> model
    """
    if not lora_path or not os.path.exists(lora_path):
        return "DefaultModel"
    
    # 规范化路径
    lora_path = os.path.normpath(lora_path)
    
    # 方法1: 匹配 /folder/ckpt-数字 格式
    match = re.search(r'/([^/]+)/ckpt-(\d+)', lora_path)
    if match:
        folder_name = match.group(1)
        epoch_num = int(match.group(2))
        return f"{folder_name}-epoch{epoch_num}"
    
    # 方法2: 匹配 /folder/ckpt.safetensors 格式
    match = re.search(r'/([^/]+)/ckpt\.safetensors$', lora_path)
    if match:
        return match.group(1)
    
    # 方法3: 匹配 ./folder/ckpt.safetensors 格式
    match = re.search(r'[./]*([^/]+)/ckpt\.safetensors$', lora_path)
    if match:
        return match.group(1)
    
    # 方法4: 从文件名中提取（兜底方案）
    filename = os.path.basename(lora_path)
    return filename.split('.')[0] if '.' in filename else filename


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '21256')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def main_worker(rank, world_size, args, dataset_configs):
    """每个进程的主函数"""
    from infer.inference import ImageGenerator
    from infer.inner_evaluation import evaluation_depth_custom_parallel

    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"[main_worker] 开始加载 pipeline, device={device}, datasets={list(dataset_configs.keys())}", flush=True)

    pipeline = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258-FP8.safetensors" if args.quantized else "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=args.qwen2vl_model_path,
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        lora=args.lora,
        device=str(device),
        args=args,
    )

    if rank == 0:
        print(f"Successfully loading pipeline from {args.model_path}.", flush=True)

    test_data_dir = resolve_eval_data_root(args, "configs")

    # 使用新的模型标识符提取函数
    model_identifier = extract_model_identifier(args.lora)

    if rank == 0:
        print(f"模型标识符: {model_identifier}", flush=True)

    all_dataset_results = {}
    aligment_map = {"depth": "least_square", "disp": "least_square_disparity", "ln": "log_space"}
    for dataset_name, config_path in dataset_configs.items():
        # 修改输出目录结构：在数据集名称外添加模型名称层级
        eval_dir = os.path.join(args.output_dir, model_identifier, dataset_name)
        test_dataset_config = os.path.join(test_data_dir, config_path)
        alignment_type = aligment_map[args.norm_type]

        if rank == 0:
            print(f"\n开始评估数据集: {dataset_name}", flush=True)
            print(f"输出目录: {eval_dir}", flush=True)
            print("=" * 80, flush=True)

        metric_tracker_Lpred, metric_tracker_Rpred, processing_times = evaluation_depth_custom_parallel(
            rank,
            world_size,
            eval_dir,
            test_dataset_config,
            args,
            pipeline,
            test_data_dir,
            alignment=alignment_type,
            save_pred_vis=args.save_viz,
        )

        # 同步所有进程
        dist.barrier()

        # 收集两个CFG配置的结果
        gathered_metrics_Lpred = [None] * world_size
        gathered_metrics_Rpred = [None] * world_size
        gathered_times = [None] * world_size

        # 从metric_tracker获取结果字典
        metrics_dict_Lpred = metric_tracker_Lpred.result() if hasattr(metric_tracker_Lpred, 'result') else {}
        # metrics_dict_Rpred = metric_tracker_Rpred.result() if hasattr(metric_tracker_Rpred, 'result') else {}

        dist.all_gather_object(gathered_metrics_Lpred, metrics_dict_Lpred)
        # dist.all_gather_object(gathered_metrics_Rpred, metrics_dict_Rpred)
        dist.all_gather_object(gathered_times, processing_times)

        if rank == 0:
            metrics_dict_Lpred, dataset_times = collect_and_merge_dual_cfg_results(rank, world_size, gathered_metrics_Lpred, gathered_times)

            if metrics_dict_Lpred:
                # 格式化并输出结果表格
                formatted_output = format_dual_cfg_results_table(dataset_name, model_identifier, metrics_dict_Lpred, dataset_times)

                print(formatted_output)
                print(f"结果保存至: {eval_dir}")

                # 存储结果用于后续汇总
                all_dataset_results[dataset_name] = {'metrics_Lpred': metrics_dict_Lpred, 'formatted_output': formatted_output, 'eval_dir': eval_dir, 'processing_times': dataset_times}

    if rank == 0:
        print(f"\n所有数据集评估完成! 结果保存在: {os.path.join(args.output_dir, model_identifier)}")

        # 保存所有数据集的双CFG评估结果摘要
        if all_dataset_results:
            save_dual_cfg_results_summary(os.path.join(args.output_dir, model_identifier), all_dataset_results, model_identifier)

    cleanup()

def main_worker_normal(rank, world_size, args, eval_datasets):
    """normal预测的多进程主函数"""
    from infer.inference import ImageGenerator
    from infer.inner_evaluation import evaluation_normal_custom_parallel

    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    pipeline = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258-FP8.safetensors" if args.quantized else "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=args.qwen2vl_model_path,
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        lora=args.lora,
        device=str(device),
        args=args,
    )

    if rank == 0:
        print(f"Successfully loading pipeline from {args.model_path}.")

    test_data_dir = resolve_eval_data_root(args, "dsine_eval")
    dataset_split_path = os.path.join(REPO_ROOT, "infer", "dataset_normal")
    
    # 使用新的模型标识符提取函数
    model_identifier = extract_model_identifier(args.lora)
    
    # 修改输出目录结构：在任务名称外添加模型名称层级
    eval_dir = os.path.join(args.output_dir, model_identifier, args.task_name)

    if rank == 0:
        print(f"模型标识符: {model_identifier}")
        print(f"输出目录: {eval_dir}")

    if rank == 0:
        print(f"\n开始并行Normal评估，使用{world_size}个GPU")
        print("=" * 80)

    # 调用并行评估函数
    all_normal_errors, all_processing_times, all_dataset_metrics = evaluation_normal_custom_parallel(
        rank, world_size, eval_dir, test_data_dir, dataset_split_path, pipeline, args, eval_datasets, save_pred_vis=args.save_viz
    )

    # 同步所有进程
    dist.barrier()

    # 收集所有GPU的结果
    gathered_normal_errors = [None] * world_size
    gathered_processing_times = [None] * world_size
    gathered_dataset_metrics = [None] * world_size

    dist.all_gather_object(gathered_normal_errors, all_normal_errors)
    dist.all_gather_object(gathered_processing_times, all_processing_times)
    dist.all_gather_object(gathered_dataset_metrics, all_dataset_metrics)

    if rank == 0:
        # 合并所有GPU的结果
        final_results = {}
        
        for dataset_name, _ in eval_datasets:
            print(f"\n合并数据集 {dataset_name} 的结果...")
            
            # 合并normal errors
            all_errors_for_dataset = []
            all_times_for_dataset = []
            
            for gpu_errors, gpu_times in zip(gathered_normal_errors, gathered_processing_times):
                if gpu_errors[dataset_name] is not None:
                    all_errors_for_dataset.append(gpu_errors[dataset_name])
                if gpu_times[dataset_name]:
                    all_times_for_dataset.extend(gpu_times[dataset_name])
            
            # 计算最终指标
            if all_errors_for_dataset:
                combined_errors = torch.cat(all_errors_for_dataset, dim=0)
                from infer.util import normal_utils
                final_metrics = normal_utils.compute_normal_metrics(combined_errors)
                
                print(f"数据集 {dataset_name} 最终结果:")
                print("mean median rmse 5 7.5 11.25 22.5 30")
                print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                    final_metrics['mean'], final_metrics['median'], final_metrics['rmse'], 
                    final_metrics['a1'], final_metrics['a2'], final_metrics['a3'], 
                    final_metrics['a4'], final_metrics['a5']
                ))
                
                final_results[dataset_name] = {
                    'metrics': final_metrics,
                    'processing_times': all_times_for_dataset,
                    'sample_count': len(combined_errors)
                }
                
                # 保存结果到文件
                dataset_output_dir = os.path.join(eval_dir, dataset_name)
                os.makedirs(dataset_output_dir, exist_ok=True)
                
                from tabulate import tabulate
                eval_text = f"Evaluation metrics for {dataset_name}:\n"
                eval_text += f"Total samples: {len(combined_errors)}\n"
                eval_text += f"Average processing time: {np.mean(all_times_for_dataset):.2f}s\n"
                eval_text += tabulate([list(final_metrics.keys()), list(final_metrics.values())])
                
                save_path = os.path.join(dataset_output_dir, "eval_metrics.txt")
                with open(save_path, "w+") as f:
                    f.write(eval_text)
                
                print(f"结果已保存至: {save_path}")
            else:
                print(f"数据集 {dataset_name}: 未找到有效数据")
                final_results[dataset_name] = None

        # 保存总体结果摘要
        summary_file = os.path.join(eval_dir, f"{model_identifier}_normal_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("Normal预测多GPU并行评估结果汇总\n")
            f.write("=" * 120 + "\n\n")
            
            for dataset_name, result in final_results.items():
                if result is not None:
                    f.write(f"数据集: {dataset_name}\n")
                    f.write(f"样本数: {result['sample_count']}\n")
                    f.write(f"平均处理时间: {np.mean(result['processing_times']):.2f}s\n")
                    metrics = result['metrics']
                    f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
                    f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % (
                        metrics['mean'], metrics['median'], metrics['rmse'], 
                        metrics['a1'], metrics['a2'], metrics['a3'], 
                        metrics['a4'], metrics['a5']
                    ))
                    f.write("-" * 60 + "\n\n")
                else:
                    f.write(f"数据集: {dataset_name} - 无有效数据\n\n")
        
        print(f"\nNormal评估总结已保存至: {summary_file}")

    cleanup()
    
def main():
    args = parse_args()
    if args.seed is not None:
        seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 检查GPU数量
    world_size = min(args.num_gpus, torch.cuda.device_count())
    if world_size <= 0:
        print("错误：未检测到可用的GPU。")
        return

    # 设置多进程相关环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    print(f"即将使用 {world_size} 个GPU进行并行推理...")

    if args.task_name == 'depth':
        test_depth_dataset_configs = parse_depth_eval_datasets(args.depth_eval_datasets)
        print(f"Depth评估数据集: {list(test_depth_dataset_configs.keys())}")

        if world_size == 1:
            # 单GPU情况，直接运行
            main_worker(0, 1, args, test_depth_dataset_configs)
        else:
            # 多GPU情况，使用multiprocessing
            try:
                mp.spawn(main_worker, args=(world_size, args, test_depth_dataset_configs), nprocs=world_size, join=True)
            except Exception as e:
                print(f"多进程执行出错: {e}")
                print("尝试降级到单GPU模式...")
                # 降级到单GPU模式
                main_worker(0, 1, args, test_depth_dataset_configs)

    elif args.task_name == 'normal':
        eval_datasets = parse_normal_eval_datasets(args.normal_eval_datasets)
        print(f"Normal评估数据集: {eval_datasets}")

        if world_size == 1:
            main_worker_normal(0, 1, args, eval_datasets)
        else:
            try:
                mp.spawn(main_worker_normal, args=(world_size, args, eval_datasets), nprocs=world_size, join=True)
            except Exception as e:
                print(f"多进程执行出错: {e}")
                print("尝试降级到单GPU模式...")
                # 降级到单GPU模式
                main_worker_normal(0, 1, args, eval_datasets)
    else:
        raise ValueError(f"不支持的 task_name: {args.task_name}，仅支持 depth 或 normal")

if __name__ == '__main__':
    main()
