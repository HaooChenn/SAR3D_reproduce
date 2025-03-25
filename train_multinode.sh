# srun -p a6000_xgpan -n 1 --gres=gpu:4 --cpus-per-task=64 -w slabgpu17 torchrun --nproc_per_node=4 --nnodes=1 \
#     --rdzv-endpoint=slabgpu17:1234 --rdzv_backend=c10d train.py --depth=16 --bs=192 --ep=200 --fp16=1 --alng=1e-3 \
#     --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet
# srun -p a6000_xgpan -n 1 --gres=gpu:4 --cpus-per-task=64 -w MICL-PanXGSvr2 torchrun --nproc_per_node=4 --nnodes=1 \
#     --rdzv-endpoint=MICL-PanXGSvr2:1234 --rdzv_backend=c10d train.py --depth=16 --bs=188 --ep=200 --fp16=1 --alng=1e-3 \
#     --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset
    
srun -p a6000_xgpan -n 1 --gres=gpu:4 --cpus-per-task=64 -w slabgpu17 torchrun --nproc_per_node=4 --nnodes=1 --rdzv-endpoint=slabgpu17:3686 --rdzv_backend=c10d train.py \
  --depth=16 --bs=24 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset