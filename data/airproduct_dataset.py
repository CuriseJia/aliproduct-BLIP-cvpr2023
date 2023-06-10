import cv2
from PIL import Image
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import os


class TrainDataset(Dataset):
    def __init__(self, transform, json_path, image_path, mode) -> None:
        self.transform = transform
        self.mode = mode
        self.image_path = image_path
        self.json_path = json_path

        with open(self.json_path, 'r') as f:
            self.pair_list = json.load(f)
        
        self.img_ids = {}
        n = 0
        for ann in self.pair_list:
            img_id = ann['product']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1   
        
    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, index):
        ann = self.pair_list[index]
        image_path = self.image_path + self.pair_list[index]['product']
        context = self.pair_list[index]['caption']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # print(self.img_ids[ann['product']])

        return image, context, self.img_ids[ann['product']]
    

class ValDataset(Dataset):
    def __init__(self, transform, json_path, image_path, mode) -> None:
        self.transform = transform
        self.mode = mode
        self.image_path = image_path
        self.json_path = json_path

        with open(self.json_path, 'r') as f:
            self.pair_list = json.load(f)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        img_id = 0
        for ann in self.pair_list:
            self.image.append(ann['product'])
            self.img2txt[img_id] = []
            self.text.append(ann['caption'])
            self.img2txt[img_id]=txt_id
            self.txt2img[txt_id]=img_id
            txt_id += 1
            img_id += 1

    def __len__(self):
        return len(self.pair_list)
        # return 10
    
    def __getitem__(self, index):
        ann = self.pair_list[index]
        image_path = self.image_path + self.pair_list[index]['product']
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image) 

        return image, index


class TestDataset(Dataset):
    def __init__(self, transform, json_path, image_path) -> None:
        self.transform = transform
        self.image_path = image_path
        self.json_path = json_path

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        p = os.listdir(self.image_path)
        len_p = len(p)
        images = []

        with open(self.json_path, 'r') as f:
            self.pair_list = json.load(f)
        
        for i in tqdm(range(len_p)):
            img_path = self.image_path + p[i]
            self.image.append(img_path)

        
        
        txt_id = 0
        img_id = 0
        for ann in self.pair_list:
            # self.image.append(ann['product'])
            self.img2txt[img_id] = []
            self.text.append(ann['caption'])
            self.img2txt[img_id]=txt_id
            self.txt2img[txt_id]=img_id
            txt_id += 1
            img_id += 1

    def __len__(self):
        return len(self.pair_list)
        # return 10
    
    def __getitem__(self, index):
        context = self.pair_list[index]
        image_path = self.image_path + self.pair_list[index]['product']
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image) 

        return image, context, index
