python sample.py \
    --model DiT-XL/2 \
    --image_size 512 \
    --num_classes 9690 \
    --seed 385 \
    --num_sampling_steps 500 \
    --ckpt /workspace/mnt/storage/zhaozhijian/checkpoints-fs2/ft/001-DiT-XL-2/epoch19-global_step2480/model \
    --text_encoder /workspace/mnt/storage/zhaozhijian/suprellm/OpenDiT/clip \
    --prompt   优雅长袖 \
    --use_textembed
    #--ckpt /workspace/mnt/storage/zhaozhijian/checkpoints-fs/005-DiT-XL-2/epoch19-global_step363300/model \

