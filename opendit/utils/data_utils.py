import os
import random
from typing import Iterator, Optional,List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

from opendit.utils.pg_utils import ProcessGroupManager


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    add_sampler=True,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    pg_manager: Optional[ProcessGroupManager] = None,
    **kwargs,
):
    r"""
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.


    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()
    if add_sampler :
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=pg_manager.size(pg_manager.dp_axis),
            rank=pg_manager.coordinate(pg_manager.dp_axis),
            shuffle=shuffle,
        )
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class VideoDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        file_list = os.listdir(data_path)
        self.video_list = [v for v in file_list if v.endswith(".npy")]

        # if toy dataset, repeat the video list
        if set(self.video_list) == set(["art-museum.npy", "lagos.npy", "man-on-the-cloud.npy", "suv-in-the-dust.npy"]):
            print("Using toy dataset, repeating the video data 100 times")
            self.video_list = self.video_list * 100
        else:
            raise ValueError("Invalid dataset")

        self.num_samples = len(self.video_list)
        # TODO: add label
        self.label_list = [random.randint(0, 9) for _ in self.video_list]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.tensor(np.load(os.path.join(self.data_path, self.video_list[idx])))
        label = torch.tensor(self.label_list[idx])
        return video, label


from transformers.modeling_utils import PreTrainedModel
def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None
) -> List[str]:
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head"]
    module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, linear_cls)
            and not any([output_layer in name for output_layer in output_layer_names])
        ):
            module_names.add(name.split(".")[-1])

    print("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)
