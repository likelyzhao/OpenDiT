cp ../sources.list /etc/apt/sources.list
apt-get update && apt-get install python3-dev -y

/workspace/mnt/storage/zhaozhijian/wudao/envpy3/bin/torchrun --standalone --nproc_per_node=8 train.py \
    --model DiT-XL/2 \
    --batch_size 4 \
    --plugin ddp \
    --mixed_precision fp32 \
    --outputs /workspace/mnt/storage/zhaozhijian/checkpoints-fs2/ \
    --data_path /workspace/mnt/storage/zhaozhijian/wudao/jd_10K_tf \
    --num_samples 141920 \
    --image_size 512 \
    --epochs 200 \
    --ckpt_every 20 \
    --enable_flashattn \
    --num_classes 9690
