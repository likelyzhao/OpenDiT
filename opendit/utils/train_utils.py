from collections import OrderedDict

import torch
import torch.distributed as dist
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer

from opendit.models.dit import DiT
from opendit.models.latte import Latte
from peft.peft_model import PeftModel


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    if not (isinstance(model, DiT) or isinstance(model, Latte)):
        model = model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            if name not in ema_params:
                #print("missing")
                name = name.replace(".module","")
                #@import pdb 
                #pdb.set_trace()
                if name in ema_params:
                    ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
                #ema_params[name] = param_data
            else:
                ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32 and isinstance(optimizer, LowLevelZeroOptimizer):
                param_id = id(param)
                master_param = optimizer._param_store.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
