# FE2E: From Editor to Dense Geometry Estimator

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://amap-ml.github.io/FE2E/)
[![Paper](https://img.shields.io/badge/arXiv-2509.04338-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.04338)
[![GitHub](https://img.shields.io/github/stars/AMAP-ML/FE2E?style=social)](https://github.com/AMAP-ML/FE2E)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/models?search=amap-ml/fe2e)


[Jiyuan Wang](https://wangjiyuan9.github.io/)<sup>1,2</sup>,
[Chunyu Lin](https://scholar.google.com/citations?hl=zh-CN&user=t8xkhscAAAAJ)<sup>1&#9993;</sup>,
[Lei Sun](https://scholar.google.com/citations?user=your-id)<sup>2&#10013;</sup>,
[Rongying Liu](https://scholar.google.com/citations?user=your-id)<sup>1</sup>,
[Mingxing Li](https://scholar.google.com/citations?user=-pfkprkAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>,
[Lang Nie](https://scholar.google.com/citations?hl=zh-CN&user=vo__egkAAAAJ)<sup>3</sup>,
[Kang Liao](https://kangliao929.github.io/)<sup>4</sup>,
[Xiangxiang Chu](https://cxxgtxy.github.io/)<sup>2</sup>,
[Yao Zhao](https://faculty.bjtu.edu.cn/5900/)<sup>1</sup>

<span class="author-block"><sup>1</sup>Beijing Jiaotong University</span>
<span class="author-block"><sup>2</sup>Alibaba Group</span>
<span class="author-block"><sup>3</sup>Chongqing University of Posts and Telecommunications</span>
<span class="author-block"><sup>4</sup>Nanyang Technological University</span><br>
<span class="author-block">
    <sup>&#9993;</sup>Corresponding author.
    <sup>&#10013;</sup>Project leader.
</span>

<br>

![teaser](https://github.com/AMAP-ML/FE2E/raw/main/demo.png)

We present **FE2E**, a DiT-based foundation model for monocular dense geometry prediction. We pioneer the adaptation of an advanced image editing model for dense geometry prediction, revealing that editing models possess inherent structural priors beneficial for these tasks. With limited supervision (71K images), **FE2E** achieves significant performance improvements in zero-shot depth and normal estimation.

![teaser2](https://github.com/AMAP-ML/FE2E/raw/main/pipeline.png)

## 📢 News
- **[2025-09-05]**: Paper released on [arXiv](https://arxiv.org/abs/2509.04338).

---

## 🛠️ Setup
This code was tested on Ubuntu 20.04, Python 3.10, and CUDA 12.1.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AMAP-ML/FE2E.git
    cd FE2E
    ```

2.  **Install dependencies:** We recommend using `conda` for environment management.
    ```bash
    Dependencies will release soon.
    ```

---

## 🔥 Training
1.  **Initialize Accelerate Environment:**
    ```bash
    accelerate config
    ```

2.  **Prepare Training Data:**
    Please refer to our paper for details on the training datasets. After downloading, organize the data as specified in the configuration files and update the corresponding paths.

3.  **Run Training Script:**
    To train the FE2E model for joint depth and normal estimation, run the training script:

     Script will release soon.

---

## 🕹️ Inference

### Testing on Your Images
1.  Place your images in a directory, for example, `assets/examples`.
2.  Run the inference script. Our model jointly predicts depth and normals.

    Script will release soon.

### Evaluation on Benchmark Datasets
1.  **Prepare Benchmark Datasets:**
    -   For **depth** estimation, download the evaluation datasets provided by [Marigold](https://github.com/prs-eth/Marigold?tab=readme-ov-file#-evaluation-on-test-datasets-):
        ```bash
        cd datasets/eval/depth/
        wget -r -np -nH --cut-dirs=4 -R "index.html*" -P . [https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/)
        ```
    -   For **normal** estimation, download the evaluation datasets provided by [DSINE](https://github.com/baegwangbin/DSINE?tab=readme-ov-file#getting-started) (`dsine_eval.zip`) into `datasets/eval/normal/` and unzip it.

2.  **Run Evaluation Script:**
    Execute the evaluation script to assess the model's performance on the prepared benchmarks:

    Script will release soon.

---

## 🤗 Model Zoo
Below are the official models released. Our framework jointly estimates depth and normals in a single forward pass.

Model will release soon.

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
