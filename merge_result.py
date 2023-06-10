import numpy as np
import os
from tqdm import tqdm

def get_keys(s):
    start_epoch = int((s.split('_')[1]).split('-')[0])
    return start_epoch

np_list = os.listdir('result/')
np_list = sorted(np_list, key=get_keys)
# print(np_list)
result=[]

for i in tqdm(range(len(np_list))):
    path = 'result/' + np_list[i]
    temp = np.load(path)
    result.append(temp)

# print(result)
res = np.vstack(result)
print(res.shape)
np.save('result.npy',res)