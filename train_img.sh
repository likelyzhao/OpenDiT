cp ../sources.list /etc/apt/sources.list
apt-get update && apt-get install python3-dev -y

/workspace/mnt/storage/zhaozhijian/wudao/envpy3/bin/torchrun --standalone --nproc_per_node=8 train.py \
    --model DiT-XL/2 \
    --batch_size 4 \
    --plugin ddp \
    --mixed_precision fp32 \
    --outputs /workspace/mnt/storage/zhaozhijian/checkpoints-fs2/ \
    --data_path /workspace/mnt/storage/zhaozhijian/wudao/jd_10K_tf \
    --load /workspace/mnt/storage/zhaozhijian/checkpoints-fs2/029-DiT-XL-2/epoch499-global_step2217500/model \
    --lr 1e-2 \
    --num_samples 141920 \
    --image_size 512 \
    --epochs 600 \
    --ckpt_every 20 \
    --num_classes 9690
