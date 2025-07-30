# NeRF Version
torchrun --nproc_per_node=1 --nnodes=1 \
  --rdzv-endpoint=localhost:2980 --rdzv_backend=c10d test.py \
  --depth=24 --fp16=2  \
  --vqvae_pretrained_path ./checkpoints/vqvae-ckpt.pt \
  --ar_ckpt_path ./checkpoints/image-condition-ckpt.pth \
  --save_path ./eval \
  --flexicubes False \
  --test_image_path ./test_files/test_images

# Flexicubes Version
# torchrun --nproc_per_node=1 --nnodes=1 \
#   --rdzv-endpoint=localhost:2980 --rdzv_backend=c10d test.py \
#   --depth=24 --fp16=2  \
#   --vqvae_pretrained_path ./checkpoints/vqvae-flexicubes-ckpt.pt \
#   --ar_ckpt_path ./checkpoints/image-condition-ckpt.pth \
#   --save_path ./eval \
#   --flexicubes True \
#   --test_image_path ./test_files/test_images
