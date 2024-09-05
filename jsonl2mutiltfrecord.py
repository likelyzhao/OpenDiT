import tfrecord
import os
from tqdm import tqdm
import json
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset


ori_path = "/workspace/mnt/storage/zhaozhijian/wudao/vehicle_plate_dataset_v1.9/plate_jsons/train_recognition.json"
out_path = "/workspace/mnt/storage/zhaozhijian/wudao/plate_HQ/"
imgroot = "/workspace/mnt/storage/zhaozhijian/wudao/vehicle_plate_dataset_v1.9/plate_number_attr/train"

if not os.path.exists(out_path):
    os.mkdir(out_path)



if 1:
    numgpu = 16 
    fidlist = []
    for i in range(numgpu):
        writer = tfrecord.TFRecordWriter(os.path.join(out_path,"data" +str(i)+".tfrecord"))
# f = open(os.path.join(out_path, str(i) + ".txt" ), "w")
        fidlist.append(writer)
   # writer = tfrecord.TFRecordWriter(os.path.join(out_path,"data.tfrecord"))
   # files = os.listdir(ori_path)
    count = 0
    total = 0
    # with open(ori_path) as f:
        
    #     for line in tqdm(f.readlines()):
    #         dict_ = json.loads(line)
    #         filename = os.path.join(imgroot, dict_["url"])


    #         from PIL import Image
    #         import numpy as np
    #         im = Image.open(filename)
    #         width, height = im.size
    #         if width < 128 or height < 128:
    #             continue
    #         if  "attribute" not in dict_.keys():
    #             continue
    #         total +=1    
          
    # total = total//numgpu* numgpu

    total = 123296 
    #total = 32000 
   
    with open(ori_path) as f:
        for line in tqdm(f.readlines()):
            dict_ = json.loads(line)
            filename = os.path.join(imgroot, dict_["url"])


            from PIL import Image
            import numpy as np
            im = Image.open(filename)
            width, height = im.size
            if width < 128 or height < 128:
                continue
            a = np.asarray(im)
    

            #import cv2
            #img = cv2.imread(filename)
            imgmat = a.tobytes()
           
            # imgmat = bytearray(os.path.getsize(filename))
            # with open(filename, 'rb') as fb:
            #     fb.readinto(imgmat)
            # print(imgmat)
            if  "attribute" not in dict_.keys():
                continue

            height, width, channels = a.shape 
            dict_["label"] = dict_["label"].replace("#","")

            if len(dict_["attribute"])==0:
                textstr = "未知类型" + dict_["label"]
            else:
                textstr = dict_["attribute"] + dict_["label"]
            fidlist[count%numgpu].write(
                {"text":(textstr.encode('utf-8'), "byte"),
                 "img":(imgmat, "byte"),
                 "width":(width,"int"),
                 "height":(height,"int"),
                 "channels":(channels,"int"),
                 }
                )
            # temp = textstr.encode('utf-8')
            #fidlist[count%numgpu].write({"text":(textstr.encode('utf-8'), "byte")})

            count +=1
            if count == total:
                break

    for writer in fidlist:
        writer.close()

    os.system("python3 -m tfrecord.tools.tfrecord2idx " + os.path.join(out_path))

tfrecord_path = os.path.join(out_path,"data{}.tfrecord")
index_path = os.path.join(out_path,"data{}.tfindex")
splits = {
    "1": 1,
}
description = {"text": "byte", "img": "byte",  "width":"int","height":"int","channels":"int"}
dataset = MultiTFRecordDataset(tfrecord_path, index_path, splits, description, infinite=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)



tfrecord_path = os.path.join(out_path,"data1.tfrecord")
index_path = None
description = {"text": "byte"}
dataset2 = TFRecordDataset(tfrecord_path, index_path, description, shuffle_queue_size=1)
loader2 = torch.utils.data.DataLoader(dataset2, batch_size=1)
iter2 = iter(loader2)


for item in loader:
    import numpy as np 
    print(item['text'][0].decode('utf-8'))
    img = np.frombuffer(item['img'][0], dtype=np.uint8).reshape(item['height'][0],item['width'][0],item['channels'][0] )
    img = Image.fromarray(img)
    
    img.save("test.jpg") 
# import cv2 
   # cv2.imwrite("test.jpg", img)

    print(np.frombuffer(item['img'], dtype=np.int8))
    print(next(iter2)['text'].decode('utf-8'))