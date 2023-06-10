import os
import json
import numpy as np
from tqdm import tqdm

def get_start_epoch(filename):
    start_epoch = int((filename.split('_')[1]).split('-')[0])
    return start_epoch

def merge_topk5_top10(npy_dir=None, npy_format=None):
    # npy_dir = os.path.abspath(npy_dir)
    # assert os.path.exists(npy_dir), 'npy_dir does not exist'
    
    # list_dir = os.listdir(npy_dir)
    # list_npy_name = []
    # for file in list_dir:
    #     if file.startswith(npy_format) and file.endswith('.npy'):
    #         list_npy_name.append(file)
    
    # list_npy_name = sorted(list_npy_name, key=get_start_epoch)
    
    # img2txt_score_list = []
    # for npy_file in list_npy_name:
    #     img2txt_score = np.load(os.path.join(npy_dir, npy_file))
    #     assert img2txt_score.shape == (1000, 20000), 'shape of npy file is not correct'
    #     img2txt_score_list.append(img2txt_score)
    # img2txt_score_matric = np.vstack(img2txt_score_list)
    # print('img2txt_score_matric shape is {}'.format(img2txt_score_matric.shape))
    img2txt_score_matric = np.load('result.npy')
    
    # temp = np.argsort(img2txt_score_matric, axis=0)
    # np.save('result_sort.npy',temp)
    temp = np.load('result_sort.npy')
    print('temp shape is {}'.format(temp.shape))
    # print(temp[0][0])
    # print()
    len_txt = img2txt_score_matric.shape[1]
    top5_indices = temp[49999:49994:-1]
    top5_values = img2txt_score_matric[top5_indices, np.arange(len_txt)]
    top10_indices = temp[49999:49989:-1]
    top10_values = img2txt_score_matric[top10_indices, np.arange(len_txt)]
    
    return top5_indices.T, top10_indices.T

def get_result_txt(img_path=None, cap_path=None, output_path=None, npy_dir=None, npy_format=None):
    assert img_path is not None, 'image path can not be None'
    assert cap_path is not None, 'caption path can not be None'
    assert output_path is not None, 'output path can not be None'
    assert npy_dir is not None, 'Para npy_dir can not be None'
    assert npy_format is not None, 'Para npy_format can not be None'
    
    img_path = os.path.abspath(img_path)
    cap_path = os.path.abspath(cap_path)
    output_path = os.path.abspath(output_path)
    
    img_list = os.listdir(img_path)
    assert len(img_list) == 50000, 'image numbers should be 50000 but {}'.format(len(img_list))
    cap_list = []
    with open(cap_path, 'r') as f:
        pair_list = json.load(f)
    for ann in pair_list:
        cap_list.append(ann['caption'])
    assert len(cap_list) == 20000, 'caption numbers should be 20000 but {}'.format(len(cap_list))
    
    print('begin merge')
    top5_indices, top10_indices = merge_topk5_top10(npy_dir=npy_dir, npy_format=npy_format)
    assert top5_indices.shape == (20000, 5), 'result top5 should be (20000, 5) but {}'.format(top5_indices.shape)
    assert top10_indices.shape == (20000, 10), 'result top10 should be (20000, 10) but {}'.format(top10_indices.shape)
    print('finish merge')
    result_top5 = []
    result_top10 = []
    
    data_list = []
    
    for i in tqdm(range(20000)):
        for idx in top5_indices[i]:
            result_top5.append(img_list[idx])
        for idx in top10_indices[i]:
            result_top10.append(img_list[idx])

        data = {
            'caption': cap_list[i],
            'top5': str(result_top5),
            'top10': str(result_top10)
        }
        data_list.append(data)

        result_top5.clear()
        result_top10.clear()
        
    with open(output_path, "w") as file:
        json.dump(data_list, file)


if __name__ == "__main__":
    get_result_txt(img_path='../airproduct/test_imgs/', 
                   cap_path='../airproduct/test_captions.json', 
                   output_path='../airproduct/output.json', 
                   npy_dir='result/', npy_format='result_')