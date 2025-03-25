import os
import shutil
from PIL import Image

# 源文件夹路径
source_path = "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/"
destination_path = "/mnt/slurm_home/ywchen/projects/VAR-image/extracted_images/"

# 初始化目标路径：删除并重新创建
if os.path.exists(destination_path):
    shutil.rmtree(destination_path)
os.makedirs(destination_path, exist_ok=True)

def extract_images_and_create_gif():
    total_extracted_images = []  # 用于存储所有文件夹提取的图片路径

    for folder in range(1, 15):  # 遍历文件夹2到98
        folder_path = os.path.join(source_path, str(folder))
        if not os.path.exists(folder_path):
            print(f"Folder {folder} does not exist. Skipping.")
            continue

        # 获取文件夹内所有图片并按编号排序
        images = sorted(
            [img for img in os.listdir(folder_path) if img.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        if not images or len(images) < 40:
            print(f"Folder {folder} does not have enough images. Skipping.")
            continue

        # 根据规则提取图片编号范围
        start_index = (folder - 2) * 10 % 24  # 起始编号，按24循环
        extracted_images = []
        for i in range(10):  # 提取10张图片
            img_index = (start_index + i) % 24  # 超过23从0开始
            img_name = f"1_{img_index}.png"
            img_path = os.path.join(folder_path, img_name)
            if os.path.exists(img_path):
                # 为图片添加前缀并复制到目标路径
                new_img_name = f"{folder}_{img_name}"
                new_img_path = os.path.join(destination_path, new_img_name)
                shutil.copy(img_path, new_img_path)
                extracted_images.append(new_img_path)
                total_extracted_images.append(new_img_path)  # 添加到总的提取列表

        # 10号文件夹额外提取16张图片
        if folder == 10:
            for i in range(10, 38):  # 再提取16张图片
                img_index = (start_index + i) % 24  # 超过23从0开始
                img_name = f"1_{img_index}.png"
                img_path = os.path.join(folder_path, img_name)
                if os.path.exists(img_path):
                    # 为图片添加前缀并复制到目标路径
                    new_img_name = f"{folder}_{img_name}"
                    new_img_path = os.path.join(destination_path, new_img_name)
                    shutil.copy(img_path, new_img_path)
                    extracted_images.append(new_img_path)
                    total_extracted_images.append(new_img_path)  # 添加到总的提取列表

        # 为当前文件夹生成独立的GIF
        gif_images = [Image.open(img) for img in extracted_images if os.path.exists(img)]
        if gif_images:
            folder_gif_path = os.path.join(destination_path, f"folder_{folder}.gif")
            gif_images[0].save(
                folder_gif_path, save_all=True, append_images=gif_images[1:], duration=100, loop=0
            )
            print(f"GIF for folder {folder} saved at {folder_gif_path}.")
        else:
            print(f"No images extracted to create a GIF for folder {folder}.")

    # 生成总的GIF
    total_gif_images = [Image.open(img) for img in total_extracted_images if os.path.exists(img)]
    if total_gif_images:
        total_gif_path = os.path.join(destination_path, "total_combined.gif")
        total_gif_images[0].save(
            total_gif_path, save_all=True, append_images=total_gif_images[1:], duration=100, loop=0
        )
        print(f"Total combined GIF saved at {total_gif_path}.")
    else:
        print("No images extracted to create the total combined GIF.")

# 执行程序
extract_images_and_create_gif()
