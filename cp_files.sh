#!/bin/bash

# 源目录和目标目录
SOURCE_DIR="/mnt/slurm_home/ywchen/projects/VAR-image/VAR"
DEST_DIR="/mnt/slurm_home/ywchen/projects/VAR-image-speed"

# 使用 rsync 进行复制，同时排除指定的文件和文件夹
rsync -av --exclude='**/__pycache__/' \
          --exclude='**/.ipynb_checkpoints/' \
          --exclude='.DS_Store' \
          --exclude='.idea/' \
          --exclude='.vscode/' \
          --exclude='llava/' \
          --exclude='_vis_cached/' \
          --exclude='_auto_*' \
          --exclude='ckpt/' \
          --exclude='log/' \
          --exclude='tb*/' \
          --exclude='img*/' \
          --exclude='local_output*' \
          --exclude='sample_*' \
          --exclude='*.pth' \
          --exclude='*.pth.tar' \
          --exclude='*.ckpt' \
          --exclude='*.log' \
          --exclude='*.txt' \
          --exclude='*.ipynb' \
          --exclude='*.locks*' \
          --exclude='*pretrained*' \
          --exclude='eval*' \
          --exclude='*sample_' \
          --exclude='*baseline*' \
          "$SOURCE_DIR/" "$DEST_DIR"

echo "复制完成，已排除指定文件和文件夹。"