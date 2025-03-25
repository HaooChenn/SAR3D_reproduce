#!/bin/bash

# 定义路径
DIR="/nas/shared/public/yslan/cyw/ln3diff_VAR/log/VAR_Image/local_output_test_offline_latent_depth_24_resume"

# 遍历匹配 ar-ckpt-*.pth 的文件
for file in "$DIR"/ar-ckpt-*.pth; do
    # 提取文件名中的数字部分
    num=$(echo "$file" | grep -oP '(?<=ar-ckpt-)\d+(?=.pth)')
    
    # 检查是否是10的倍数
    if (( num % 10 != 0 )); then
        # 如果不是10的倍数，则删除文件
        echo "Deleting $file"
        rm "$file"
    fi
done
