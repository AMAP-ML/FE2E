# FE2E: From Editor to Dense Geometry Estimator

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://amap-ml.github.io/FE2E/)
[![Paper](https://img.shields.io/badge/arXiv-2509.04338-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.04338)
[![GitHub](https://img.shields.io/github/stars/AMAP-ML/FE2E?style=social)](https://github.com/AMAP-ML/FE2E)

![teaser](assets/demo.png)

This repository currently provides the FE2E inference and benchmark-evaluation package built on top of Step1X-Edit. It is intended for reproducing the released depth/normal evaluation workflow, while keeping large datasets and model checkpoints outside git.

![pipeline](assets/pipeline.png)

## 📢 News

- **[2026-03-17]**: Inference and evaluation code are organized for public use, with benchmark configs/splits and example run logs.
- **[2025-09-05]**: Paper released on [arXiv](https://arxiv.org/abs/2509.04338).

---

## 📦 What Is Released

Currently included in this repository:

- FE2E evaluation entrypoint: `evaluation.py`
- FE2E inference entrypoint: `infer/inference.py`
- benchmark dataset loaders, configs, and split files
- README and example logs under `logs/`

Not included in git:

- benchmark dataset payloads
- Step1X-Edit base model checkpoints
- FE2E LoRA checkpoint files
- training code

---

## 🛠️ Setup

This code is expected to run in a Python environment with the dependencies from:

```bash
pip install -r requirements.txt
```

Useful runtime flags already supported in this repo:

- `--eval_data_root`: explicit benchmark data root for `evaluation.py`
- `--empty_prompt_cache`: cache path for empty-prompt inference in `infer/inference.py`
- `MASTER_PORT`: can be set externally to avoid port conflicts across jobs

Supported benchmarks:

- Depth: `nyu_v2`, `kitti`, `eth3d`, `diode`, `scannet`
- Normal: `nyuv2`, `scannet`, `ibims`, `sintel`, `oasis`, `hypersim`

---

## 🔥 Training

- [ ] Training code is not included in this release yet.

---

## 🕹️ Inference and Evaluation

### External Weights and Data

This repo keeps only code. We recommend mounting weights and benchmark data into the following paths:

```text
FE2E/
├── pretrain/
│   ├── step1x-edit-i1258.safetensors
│   ├── step1x-edit-v1p1-official.safetensors
│   └── vae.safetensors
├── lora/
│   └── LDRN.safetensors
├── infer/
│   ├── eth3d/
│   │   └── eth3d.tar
│   └── dsine_eval/
│       ├── nyuv2/
│       └── scannet/
└── logs/
```

In the current local setup, these paths are mounted via symlinks so the repository itself stays lightweight.

### Model Sources

- Step1X-Edit base weights should come from the official [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) release.
- FE2E evaluation currently loads the FE2E LoRA checkpoint through `--lora`.
- This repository does not redistribute model weights inside git.

### Benchmark Data Sources

- Depth benchmarks follow the external evaluation data convention used by [Marigold](https://github.com/prs-eth/Marigold).
- Normal benchmarks follow the external evaluation data convention used by [DSINE](https://github.com/baegwangbin/DSINE).
- This repository does not redistribute benchmark data inside git.

### Run Evaluation

`ScanNet normal`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=21258 \
PYTHONUNBUFFERED=1 \
python -u evaluation.py \
  --model_path ./pretrain \
  --eval_data_root ./infer \
  --output_dir ./infer/eval_verify_scannet_normal_8gpu \
  --num_gpus 8 \
  --num_samples -1 \
  --lora ./lora/LDRN.safetensors \
  --single_denoise \
  --prompt_type empty \
  --norm_type ln \
  --task_name normal \
  --normal_eval_datasets scannet
```

Expected result:

- `mean ≈ 13.8166`
- `11.25° ≈ 67.2134`

`ETH3D depth`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=21257 \
PYTHONUNBUFFERED=1 \
python -u evaluation.py \
  --model_path ./pretrain \
  --eval_data_root ./infer \
  --output_dir ./infer/eval_verify_eth3d_8gpu \
  --num_gpus 8 \
  --num_samples -1 \
  --lora ./lora/LDRN.safetensors \
  --single_denoise \
  --prompt_type empty \
  --norm_type ln \
  --task_name depth \
  --depth_eval_datasets eth3d
```

Expected result:

- `abs_rel ≈ 0.0379`
- `sq_rel ≈ 0.0520`
- `rmse ≈ 0.4701`
- `rmse_log ≈ 0.0546`
- `a1 ≈ 0.9877`

For multiple datasets, pass a comma-separated list to `--depth_eval_datasets` or `--normal_eval_datasets`.

---

## 🧾 Example Logs

The `logs/` directory contains successful 8-GPU verification logs that show normal per-sample progress and final metric summaries:

- `logs/verify_scannet_normal_8gpu_20260317_171345.log`
- `logs/verify_eth3d_8gpu_20260317_172004.log`

Verified summaries:

- ScanNet normal: `mean = 13.8166`, `11.25° = 67.2134`
- ETH3D depth: `abs_rel = 0.0379`, `sq_rel = 0.0520`, `rmse = 0.4701`, `rmse_log = 0.0546`, `a1 = 0.9877`, `a2 = 0.9966`, `a3 = 0.9985`

These logs are included to show what a normal run looks like after the code and paths are set up correctly.

---

## 🤗 Release Status

- Inference/evaluation code: released in this repository
- Benchmark configs/splits: released in this repository
- Example verification logs: released in this repository
- Training code: not released here yet
- Model weights and benchmark datasets: external assets, not tracked in git

---

## 🎓 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{wang2025editor,
  title={From Editor to Dense Geometry Estimator},
  author={Wang, JiYuan and Lin, Chunyu and Sun, Lei and Liu, Rongying and Nie, Lang and Li, Mingxing and Liao, Kang and Chu, Xiangxiang and Zhao, Yao},
  journal={arXiv preprint arXiv:2509.04338},
  year={2025}
}
```
