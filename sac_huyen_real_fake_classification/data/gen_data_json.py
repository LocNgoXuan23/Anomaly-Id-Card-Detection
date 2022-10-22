import os
import json
import random
from tqdm import tqdm

SPLIT = 0.8

def write_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def gen_paths(ROOT, class_name):
    img_paths = os.listdir(ROOT)
    data = []
    for i in tqdm(img_paths):
        tmp = os.path.join('..', 'data', ROOT, i, class_name)
        data.append(tmp)
    return data

data_0 = gen_paths('huyen_sign_images_real', '0')
data_1 = gen_paths('huyen_sign_images_fake', '1')

data = data_0 + data_1
random.shuffle(data)
# print(data)

train_data = data[:int(len(data)*SPLIT)]
val_data = data[int(len(data)*SPLIT):]
print(f'LEN TRAIN : {len(train_data)}')
print(f'LEN VAL : {len(val_data)}')

write_json('train.json', train_data)
write_json('val.json', val_data)