cp ../sources.list /etc/apt/sources.list
apt-get update && apt-get install python3-dev -y

# cp dataset to local
# cp -r /workspace/mnt/storage/zhaozhijian/wudao/jd_10K_tf /dev/shm

/workspace/mnt/storage/zhaozhijian/dataset_dist/envpy3/bin/torchrun --nnodes=$WORLD_SIZE --master_addr=$MASTER_ADDR --nproc_per_node=8  --master_port=$MASTER_PORT --node_rank=$RANK  \
    train_dist.py \
    --model DiT-XL/2 \
    --batch_size 4 \
    --plugin hybrid \
    --mixed_precision bf16 \
    --outputs /workspace/mnt/storage/zhaozhijian/checkpoints-fs2/ \
    --data_path /workspace/mnt/storage/zhaozhijian/dataset_dist/jd_10K_tf \
    --num_samples 141920 \
    --num_workers 64 \
    --image_size 512 \
    --epochs 200 \
    --ckpt_every 20 \
    --num_classes 9690
