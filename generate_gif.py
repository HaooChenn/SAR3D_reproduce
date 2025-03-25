import os
from PIL import Image
import subprocess

# 源文件夹路径
source_path = "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_wild_figure/hat1730871348.0407984/"

def find_and_create_mp4():
    for root, dirs, files in os.walk(source_path):
        # 筛选包含 0.png 到 23.png 的文件夹
        required_images = [f"{i}.png" for i in range(24)]
        required_images_jpg = [f"{i}.jpg" for i in range(24)]

        if all(img in files for img in required_images):
            # 获取目标路径
            gif_path = os.path.join(root, "output.gif")
            mp4_path = os.path.join(root, "output.mp4")
            
            # 按顺序读取图片并创建 GIF
            images = [Image.open(os.path.join(root, f"{i}.png")) for i in range(24)]
            images[0].save(
                gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
            )
            print(f"GIF created at: {gif_path}")

            # 将 GIF 转换为 MP4
            convert_gif_to_mp4(gif_path, mp4_path)

        elif all(img in files for img in required_images_jpg):
            # 获取目标路径
            gif_path = os.path.join(root, "output.gif")
            mp4_path = os.path.join(root, "output.mp4")
            
            # 按顺序读取图片并创建 GIF
            images = [Image.open(os.path.join(root, f"{i}.jpg")) for i in range(24)]
            images[0].save(
                gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
            )
            print(f"GIF created at: {gif_path}")

            # 将 GIF 转换为 MP4
            # convert_gif_to_mp4(gif_path, mp4_path)


def convert_gif_to_mp4(gif_path, mp4_path):
    """
    使用 ffmpeg 将 GIF 转换为 MP4 文件
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", gif_path, "-movflags", "faststart", "-pix_fmt", "yuv420p", mp4_path],
            check=True
        )
        print(f"MP4 created at: {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting GIF to MP4: {e}")

# 执行程序
find_and_create_mp4()