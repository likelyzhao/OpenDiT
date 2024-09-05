# Modified from Meta DiT: https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import argparse
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.models.latte import Latte_models
from opendit.utils.download import find_model
from opendit.vae.reconstruct import save_sample
from opendit.vae.wrapper import AutoencoderKLWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"vae").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.use_video:
        # Wrap the VAE in a wrapper that handles video data
        # Use 3d patch size that is divisible by the input size
        vae = AutoencoderKLWrapper(vae)
        input_size = (args.num_frames, args.image_size, args.image_size)
        for i in range(3):
            assert input_size[i] % vae.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // vae.patch_size[i] for i in range(3)]
    else:
        input_size = args.image_size // 8

    dtype = torch.float32
    if "DiT" in args.model:
        if "VDiT" in args.model:
            assert args.use_video, "VDiT model requires video data"
        else:
            assert not args.use_video, "DiT model requires image data"
        model_class = DiT_models[args.model]
    elif "Latte" in args.model:
        assert args.use_video, "Latte model requires video data"
        model_class = Latte_models[args.model]
    else:
        raise ValueError(f"Unknown model {args.model}")
    model = (
        model_class(
            input_size=input_size,
            num_classes=args.num_classes,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            dtype=dtype,
            text_encoder=args.text_encoder,
            use_textembed=args.use_textembed   
        )
        .to(device)
        .to(dtype)
    )
##PEFT
    from peft import TaskType,LoraConfig,get_peft_model
    lora_config = LoraConfig(
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha * args.lora_rank,
                lora_dropout=args.lora_dropout,
                target_modules=['qkv'],
                modules_to_save=args.additional_target
            )
    model = get_peft_model(model, lora_config)
#'base_model.model.module.module.final_layer.adaLN_modulation.1.bias'
#'base_model.model.final_layer.adaLN_modulation.1.bias')

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    # new state_dict 
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace(".module","")
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    merged_model = model.merge_and_unload()
    # OrderedDict(model.named_parameters()).keys()
    import pdb 
    pdb.set_trace()
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    for i in range(100):
        generation(model, diffusion, vae , input_size, device, "sample" + str(i) + ".png")


def generation(model, diffusion, vae ,input_size, device, savepath):
    if args.use_video:
        # Labels to condition the model with (feel free to change):
        class_labels = ["Biking", "Cliff Diving", "Rock Climbing Indoor", "Punch", "TaiChi"]
        n = len(class_labels)
        z = torch.randn(n, vae.out_channels, *input_size, device=device)
        y = class_labels * 2

    else:
        if args.prompts is not None:
            class_labels = args.prompts.split(",")
            n = len(class_labels)
            z = torch.randn(n, 4, input_size, input_size, device=device)
            class_labels.append("")
            y = class_labels
            #y = class_labels*2
            #y=""
            print("y=",y )
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            z = torch.cat([z, z], 0)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample
            #save_sample(samples)
            save_image(samples, savepath , nrow=4, normalize=True, value_range=(-1, 1))
            return 

        else:
        # Labels to condition the model with (feel free to change):
            if args.num_classes == 9690:
                import random
                class_labels = [random.randint(0, 9690) for i in range(8)]
                #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
            elif args.num_classes == 1000:
                class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
            else:
                class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            n = len(class_labels)
            z = torch.randn(n, 4, input_size, input_size, device=device)
            y = torch.tensor(class_labels, device=device)
            y_null = torch.tensor([0] * n, device=device)
            y = torch.cat([y, y_null], 0)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    if args.use_video:
        samples = vae.decode(samples)
        save_sample(samples)
    else:
        samples = vae.decode(samples / 0.18215).sample
        save_image(samples, savepath , nrow=4, normalize=True, value_range=(-1, 1))

"""
    # Create sampling noise:
    if args.use_video:
        # Labels to condition the model with (feel free to change):
        class_labels = ["Biking", "Cliff Diving", "Rock Climbing Indoor", "Punch", "TaiChi"]
        n = len(class_labels)
        z = torch.randn(n, vae.out_channels, *input_size, device=device)
        y = class_labels * 2

    else:
        if args.prompts is not None:
            class_labels = args.prompts.split(",")
            n = len(class_labels)
            z = torch.randn(n, 4, input_size, input_size, device=device)
            class_labels.append("")
            y = class_labels
            #y = class_labels*2
            #y=""
            print("y=",y )
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            z = torch.cat([z, z], 0)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample
            #save_sample(samples)
            save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
            return 

        else:
        # Labels to condition the model with (feel free to change):
            if args.num_classes == 9690:
                class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
            elif args.num_classes == 1000:
                class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
            else:
                class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            n = len(class_labels)
            z = torch.randn(n, 4, input_size, input_size, device=device)
            y = torch.tensor(class_labels, device=device)
            y_null = torch.tensor([0] * n, device=device)
            y = torch.cat([y, y_null], 0)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    if args.use_video:
        samples = vae.decode(samples)
        save_sample(samples)
    else:
        samples = vae.decode(samples / 0.18215).sample
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()) + list(Latte_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images.")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--additional_target", type=str, default="")
    parser.add_argument("--use_textembed", action="store_true", help="Use use_textembed")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    args = parser.parse_args()
    main(args)
