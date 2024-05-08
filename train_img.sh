#cp ../sources.list /etc/apt/sources.list
#apt-get update && apt-get install python3-dev -y

torchrun --standalone --nproc_per_node=8 train.py \
    --model DiT-XL/2 \
    --batch_size 16 \
    --plugin ddp \
    --mixed_precision fp32 \
    --outputs /data/LLaMA-Factory/OpenDiT/save/ \
    --data_path /data/OpenDiT/plate_HQ/ \
    --num_samples 123296 \
    --image_size 256 \
    --epochs 100 \
    --ckpt_every 20 \
    --use_textembed \
    --text_encoder /data/LLaMA-Factory/OpenDiT/clip/
