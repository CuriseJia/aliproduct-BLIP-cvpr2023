from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import json
from tqdm import tqdm
import numpy as np

from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
from models.blip_itm import blip_itm

def load_image(image, image_size, device):
    images = []
    for i in tqdm(range(len(image))):
        raw_image = Image.open(image[i]).convert('RGB')

        w, h = raw_image.size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        # img = transform(raw_image).unsqueeze(0).to(device)
        img = transform(raw_image).to(device)
        images.append(img)
    
    imgs = torch.stack(images)

    return imgs

def predict(image, caption):
    assert caption is not None, 'Please type a caption for mage text matching task.'

    device = 'cuda:3'
    im = load_image(image, image_size=384, device=device)
    print('success load.')
    model = blip_itm(pretrained='../BLIP/output/epoch_16_nccl.pth' ,image_size=384, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
    model.eval()
    model = model.to(device)
    print('success model init')
    
    b,c,h,w = im.shape
    l = len(caption)
    p=0
    result = np.ndarray((15,10),dtype=float)
    for i in tqdm(range(b)):
        img_temp=im[p:p+1,:,:,:]
        p+=5
        k=0
        for j in range(0,l):
            ca_temp = caption[k:k+1]
            # print(len(ca_temp))
            k+=5
            itc_score = model(img_temp, ca_temp, match_head='itc')  # (5,5)
            
            for m in range(5):
                result[p-5][m]=itc_score[0][m]
                result[p-4][m]=itc_score[1][m]
                result[p-3][m]=itc_score[2][m]
                result[p-2][m]=itc_score[3][m]
                result[p-1][m]=itc_score[4][m]

            if k==l:
                break
        
        if p==b:
            break

    
    # print('result is {}'.format(result))
    np.save('../BLIP/itm_result.npy',result)

    result = torch.from_numpy(result)
            
    top5_values, top5_indices = torch.topk(result, k=5, dim=0)
    top10_values, top10_indices = torch.topk(result, k=10, dim=0)
    top5_indices = top5_indices.T
    top10_indices = top10_indices.T
    top5_values = top5_values.T
    top10_values = top10_values.T

    return top5_indices, top10_indices, top5_values, top10_values


if __name__=="__main__":
    images_path = '../airproduct/test_imgs/'
    p = os.listdir(images_path)
    len_p = len(p)
    cap_path = '../airproduct/test_captions.json'
    images = []
    captions = []
    for i in tqdm(range(len_p//5)):
        img_path = images_path + p[i]
        images.append(img_path)
    print(len(images))

    with open('../airproduct/test_captions.json', 'r') as f:
        pair_list = json.load(f)
    
    for ann in pair_list:
        captions.append(ann['caption'])
    for i in range((len(images)-len(captions))):
        captions.append('')
    print(len(captions))
    
    top5_indices, top10_indices, top5_values, top10_values = predict(image=images, caption=captions)
    np.save('../BLIP/top5.npy',top5_indices)
    np.save('../BLIP/top10.npy',top10_indices)
    np.save('../BLIP/top5_v.npy',top5_values)
    np.save('../BLIP/top10_v.npy',top10_values)
    print('finish')