<div align="center">

<h1>
SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE
</h1>

<p align="center">
<a href="https://cyw-3d.github.io/projects/SAR3D"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome" height=25></a>
<a href="https://arxiv.org/abs/2411.16856"><img src="https://img.shields.io/badge/arXiv-2411.16856-b31b1b?style=for-the-badge&logo=arxiv" height=25></a>
</p>

**[Yongwei Chen¹](https://cyw-3d.github.io) &nbsp;•&nbsp; [Yushi Lan¹](https://nirvanalan.github.io) &nbsp;•&nbsp; [Shangchen Zhou¹](https://shangchenzhou.com) &nbsp;•&nbsp; [Tengfei Wang²](https://tengfei-wang.github.io) &nbsp;•&nbsp; [Xingang Pan¹](https://xingangpan.github.io)**

¹S-lab, Nanyang Technological University  
²Shanghai Artificial Intelligence Laboratory

**CVPR 2025**

<div align="center">
<img src="https://github.com/user-attachments/assets/badac244-f8ee-41c2-8129-b09cf6404b91" width="800px">
</div>

</div>

## 🌟 Features
- 🔄 **Autoregressive Modeling**
- ⚡️ **Ultra-fast 3D Generation** (<1s)
- 🔍 **Detailed Understanding**


## 🛠️ Installation & Usage

### Prerequisites

We've tested SAR3D on the following environments:

<details open>
<summary><b>Rocky Linux 8.10 (Green Obsidian)</b></summary>

- Python 3.9.8
- PyTorch 2.2.2 
- CUDA 12.1
- NVIDIA H200
</details>

<details open>
<summary><b>Ubuntu 20.04</b></summary>

- Python 3.9.16
- PyTorch 2.0.0
- CUDA 11.7  
- NVIDIA A6000
</details>

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/cyw-3d/SAR3D.git
cd SAR3D
```

2. **Set up environment**
```bash
conda env create -f environment.yml
```

3. **Download pretrained models** 📥

The pretrained models will be automatically downloaded to the `checkpoints` folder during first run.

You can also manually download them from our [model zoo](https://huggingface.co/cyw-3d/sar3d):

| Model | Description | Link |
|-------|-------------|------|
| VQVAE | Base VQVAE model | [vqvae-ckpt.pt](https://huggingface.co/cyw-3d/sar3d/resolve/main/image-condition-ckpt.pth) |
| SAR3D | Image-conditioned model | [image-condition-ckpt.pth](https://huggingface.co/cyw-3d/sar3d/resolve/main/vqvae-ckpt.pt) |

4. **Run inference** 🚀

To test the model on your own images:

1. Place your test images in the `test_images` folder
2. Run the inference script:
```bash
bash test_image.sh
```

## 📚 Training

### Dataset

The dataset is available for download at [Hugging Face](https://huggingface.co/datasets/cyw-3d/sar3d-dataset).

The dataset consists of 8 splits containing preprocessed data based on [G-buffer Objaverse](https://aigc3d.github.io/gobjaverse/), including:
- Rendered images
- Depth maps 
- Camera poses
- Text descriptions
- Normal maps
- Latent embeddings

The dataset covers over 170K unique 3D objects, augmented to more than 630K data pairs. A data.json file is provided that maps object IDs to their corresponding categories.

### Training Command

This script will train the image-conditioned model on the training data in the `<DATA_DIR>` folder.

```bash
bash train_image.sh <GPU_NUM> <VQVAE_PATH> <OUT_DIR> <DATA_DIR>
```

## 📋 Roadmap

- [x] Inference and Training Code for Image-conditioned Generation
- [x] Dataset Release
- [ ] VQVAE training code
- [ ] Inference and Training Code for Text-conditioned Generation
- [ ] Code for Understanding

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{chen2024sar3d,
    title={SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE},
    author={Chen, Yongwei and Lan, Yushi and Zhou, Shangchen and Wang, Tengfei and Pan, Xingang},
    booktitle={CVPR},
    year={2025}
}