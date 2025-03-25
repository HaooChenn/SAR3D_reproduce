# d16, 256x256
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1  --node_rank=0 train.py \
 --depth=16 --bs=240 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/data/datasets/ImageNet

#debug
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1  --node_rank=0 --master_port=29501 train.py \
 --depth=16 --bs=88 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/data/datasets/ImageNet