from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import json
from tqdm import tqdm
import numpy as np
import argparse

from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
from models.blip_itm import blip_itm

def load_image(image, image_size, device, index, batch):
    # print('start load image')
    images = []
    for i in range(batch):
        raw_image = Image.open(image[index+i]).convert('RGB')

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

def predict(image, caption, args):
    assert caption is not None, 'Please type a caption for mage text matching task.'

    device = args.device
    # im = load_image(image, image_size=384, device=device)
    # print('success load.')
    model = blip_itm(pretrained=args.pretrained ,image_size=384, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
    model.eval()
    model = model.to(device)
    print('success model init')
    
    # b,_,_,_ = im.shape
    l = len(caption)
    p=0
    total = args.end - args.start
    result = np.ndarray((total,20000),dtype=float)
    total = total//args.batch
    
    for i in tqdm(range(total)):
        
        img_temp = load_image(image, 384, device, p, args.batch)

        p+=args.batch

        # print(img_temp.shape)
        for j in tqdm(range(l)):
            ca_temp = caption[j]
            itm_output = model(img_temp, ca_temp, match_head='itm')  #
            itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
            
            result[p-10][j] = itm_score[0]
            result[p-9][j] = itm_score[1]
            result[p-8][j] = itm_score[2]
            result[p-7][j] = itm_score[3]
            result[p-6][j] = itm_score[4]
            result[p-5][j] = itm_score[5]
            result[p-4][j] = itm_score[6]
            result[p-3][j] = itm_score[7]
            result[p-2][j] = itm_score[8]
            result[p-1][j] = itm_score[9]
        
        if p>=1000:
            break
    
    np.save(args.save_result, result)


def main(args):
    images_path = '/public/home/jiayanhao/airproduct/test_imgs/'
    p = os.listdir(images_path)
    cap_path = '/public/home/jiayanhao/airproduct/test_captions.json'
    images = []
    captions = []
    for i in range(args.start, args.end):
        img_path = images_path + p[i]
        images.append(img_path)

    with open(cap_path, 'r') as f:
        pair_list = json.load(f)
    
    for ann in pair_list:
        captions.append(ann['caption'])

    # top5_indices, top10_indices, top5_values, top10_values = predict(image=images, caption=captions, args=args)
    predict(image=images, caption=captions, args=args)

    # np.save(args.save_top5_indices, top5_indices)
    # np.save(args.save_top10_indices, top10_indices)
    # np.save(args.save_top5_values, top5_values)
    # np.save(args.save_top10_values, top10_values)
    
    print('finish.')

if __name__=="__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=999)
    parser.add_argument('--save_top5_indices', type=str, default='result/top5.npy')
    parser.add_argument('--save_top10_indices', type=str, default='result/top10.npy')
    parser.add_argument('--save_top5_values', type=str, default='result/top5.npy')
    parser.add_argument('--save_top10_values', type=str, default='result/top10.npy')
    parser.add_argument('--save_result', type=str, default='result/test_result.npy')
    parser.add_argument('--pretrained', type=str, default='/public/home/jiayanhao/BLIP/output/epoch_16_nccl.pth')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch', type=int, default=10)
    args = parser.parse_args()
    
    main(args)



