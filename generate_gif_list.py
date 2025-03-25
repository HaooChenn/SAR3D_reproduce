import os
from PIL import Image
import subprocess

# 源文件夹路径
# source_path = "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/51/268111/1/"
# source_path_list = # Define the list of file paths
save_path = "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/output_videos"
# source_path_list = [
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/51/268111/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/51/269024/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/7/45237/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/7/45280/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/7/47745/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/7/48573/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/7/48976/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/51/269865/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/56/293359/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/56/294895/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/121/616810/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/125/635560/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/125/635658/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/125/637641/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/125/637643/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/125/638832",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/133/675097/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/133/677860/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/133/679420/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/137/695721/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/145/736399/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/145/738132/1/",
#     "/mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57/Animals/145/738510/1/"
# ]
source_path_list = ["/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/2",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/3",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/4",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/5",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/6",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/7",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/8",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/9",
"/mnt/slurm_home/ywchen/projects/VAR-image/VAR/ten_scale/it0_slice_8_1732687657/10",]

# Extract the parent directories
# parent_directories = {os.path.dirname(path) for path in file_paths}

# # Display the result to the user
# import pandas as pd
# df = pd.DataFrame(list(parent_directories), columns=["Parent Directory"])
# import ace_tools as tools; tools.display_dataframe_to_user(name="Parent Directories", dataframe=df)

def find_and_create_mp4():
    for root, dirs, files in os.walk(source_path):
        # 筛选包含 0.png 到 23.png 的文件夹
        required_images = [f"1_{i}.png" for i in range(24)]
        required_images_jpg = [f"1_{i}.jpg" for i in range(24)]

        if all(img in files for img in required_images):
            print(root)
            # 获取目标路径
            # import ipdb; ipdb.set_trace()
            # gif_path = os.path .join(save_path, root.split("/")[-3:-1][0] + "_" + root.split("/")[-3:-1][1] + ".gif")
            gif_path = os.path.join(root, "output.gif")
            mp4_path = os.path .join(save_path, root.split("/")[-3:-1][0] + "_" + root.split("/")[-3:-1][1] + ".mp4")
            # gif_path = os.path.join(root, "output.gif")
            # mp4_path = os.path.join(root, "output.mp4")
            
            # 按顺序读取图片并创建 GIF
            # st()
            images = [Image.open(os.path.join(root, f"1_{i}.png")) for i in range(24)]
            images[0].save(
                gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
            )
            print(f"GIF created at: {gif_path}")

            # 将 GIF 转换为 MP4
            # convert_gif_to_mp4(gif_path, mp4_path)

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
    使用 ffmpeg 将 GIF 无损转换为 MP4 文件
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",  # 覆盖输出文件
                "-i", gif_path,  # 输入文件路径
                "-c:v", "libx264",  # 使用 H.264 编码
                # "-crf", "0",  # 无损质量
                "-preset", "veryslow",  # 更高的编码效率，保持质量
                "-pix_fmt", "yuv420p",  # 像素格式，兼容大多数播放器
                "-r", "15",  # 帧率保持和原始 GIF 一致（需要确认 GIF 帧率）
                "-movflags", "faststart",  # 提高 MP4 文件流式播放性能
                mp4_path,
            ],
            check=True
        )
        print(f"MP4 created at: {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting GIF to MP4: {e}")


# 执行程序
for source_path in source_path_list:
    print(f"Processing: {source_path}")
    find_and_create_mp4()