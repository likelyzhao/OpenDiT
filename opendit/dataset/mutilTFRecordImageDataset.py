from tfrecord.torch.dataset import MultiTFRecordDataset
from torch.utils.data import IterableDataset
from typing import Any, Callable, List, Optional, Tuple, Iterable
import os


class MutilTFRecordImageDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        rank: int,
        world_size :int, 
        description : dict = {},
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        self.root = root
        self.global_input_paths = [self.root + "/" + x
                                   for x in os.listdir(self.root)]
        
        files = [x for _, x in enumerate(self.global_input_paths)
                         if x.find('.tfrecord') != -1]
        self.local_input_paths = [x for i, x in
                                  enumerate(files)
                                  if i % world_size == rank]
        
        splits = {}
        for file_path in self.local_input_paths:   
            splits[file_path.replace('.tfrecord', '')] = 1.0/len(self.local_input_paths)

        tfrecord_path = "{}.tfrecord"
        index_path = "{}.tfindex"
        if not bool(description):     
            description = {"text": "byte", "img": "byte",  "width":"int","height":"int","channels":"int"}
        #description = {"label": "int", "img": "byte",  "width":"int","height":"int","channels":"int"}
        self.dataset = MultiTFRecordDataset(tfrecord_path, index_path, splits, description, infinite=False)
        self.transform = transform
        self.target_transform = target_transform
        #loader = torch.utils.data.DataLoader(dataset, batch_size=1)


    def tfrecord_iterator(
        self,
        TFDataset: MultiTFRecordDataset,
        transformer: Optional[Callable] = None,
    ) -> Iterable[dict]:
        for item in TFDataset:
            import numpy as np 
            from PIL import Image
            resdict ={}
            if 'text' in item:
                resdict["text"] = item['text'].decode('utf-8')
            if 'label' in item:
                resdict['label'] = item['label']
            
            #print(resdict["text"])
            #print(item['height'][0])
            #print(item['channels'][0])
            img = np.frombuffer(item['img'], dtype=np.uint8).reshape(item['height'][0],item['width'][0],item['channels'][0])
            try:
                img = Image.fromarray(img)
            except :
                #import pdb
                #pdb.set_trace()
                print(img.shape)
                print(len(item['img']))
                print(item['channels'][0])
                print(item["width"][0])
                print(item['height'][0])
                print(item['text'].decode('utf-8'))
            if transformer is not None:
                img = transformer(img)
            resdict["image"] = img
            yield resdict
    

    def __iter__(self):

        return self.tfrecord_iterator(self.dataset, self.transform)

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode
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
    
    def pad(pil_image):
        shape = pil_image.size
        padleft, padright, padbottom ,padtop = 0,0,0,0
        if shape[1] > shape[0]:
            padleft = (shape[1] - shape[0])//2
            padright = shape[1] - shape[0] - padleft
        else:
            padtop = (shape[0] - shape[1] )//2
            padbottom = shape[0] - shape[1] - padtop

        new_width = shape[0] + padright + padleft
        new_height = shape[1] + padtop + padbottom
        if new_height == shape[1] and new_width == shape[0]:
            return pil_image
        else:
            result = Image.new(pil_image.mode, (new_width, new_height), (0,0,0))
            result.paste(pil_image, (padleft, padtop))
            return result




    def get_transforms_image(image_size=256):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                #transforms.RandomRotation([-180,180], interpolation=InterpolationMode.BILINEAR),
                #transforms.RandomHorizontalFlip(),
                #transforms.Resize(image_size),
                #transforms.Lambda(lambda pil_image: pad(pil_image)),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                #transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        return transform

    dateset = MutilTFRecordImageDataset(root= "/workspace/mnt/storage/zhaozhijian/llm-chinese/Ecommerce_tf_small", 
                                        rank=0, world_size=1,description = {"text": "byte", "img": "byte",  "width":"int","height":"int","channels":"int"},
                                        transform=get_transforms_image(512))
    it = iter(dateset)
    for item in it:
        print(item['text'])
        print(item["image"].size)
        item["image"].save("1.jpg")
