import cv2
import sys
sys.path.insert(1, 'yolov7')
import numpy as np
from load_model import *
from engine import *

def detect_text(img, model):
    copy_img = img.copy()
    imgs = [img]
    results = model(imgs)  
    datas = results.pandas().xyxy[0].values.tolist()
    img = np.squeeze(results.render())[..., ::-1]
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    crop_imgs = []
    for data in datas:
        crop_img = copy_img[round(data[1]):round(data[3]), round(data[0]):round(data[2])]
        crop_imgs.append(crop_img)

    # img = resize(img, 50)
    return img, crop_imgs, datas

