from re import T
import cv2
import numpy as np
import os
from text_detection import detect_text
from sac_huyen_detection import detect_sac_huyen
from sac_huyen_classification_model import my_model
from sign_classification import sign_classifier
from load_model import *
from engine import *

# IS TRUE FALSE
IS_SHOW_FULL_IMAGE = True
IS_SHOW_TEXT = True
IS_SHOW_SAC_SIGN = True
IS_SHOW_HUYEN_SIGN = True

# PATH
FULL_CMND_IMG_PATH = os.path.join('data', 'mt2.jpeg')
DETECT_TEXT_MODEL_PATH = os.path.join('cktp', 'best_text_detection.pt')
DETECT_SAC_HUYEN_MODEL_PATH = os.path.join('cktp', 'best.pt')
SAC_CLASSIFICATION_MODEL_PATH = os.path.join('cktp', 'sac_classification.h5')
HUYEN_CLASSIFICATION_MODEL_PATH = os.path.join('cktp', 'huyen_classification.h5')

# DEFINE MODEL
text_detection_model = load_model(model_path=DETECT_TEXT_MODEL_PATH, conf=0.5)
sac_huyen_detection_model = load_model(model_path=DETECT_SAC_HUYEN_MODEL_PATH, conf=0.5)
sac_classification_model = my_model((224,224,1),2)
sac_classification_model.load_weights(SAC_CLASSIFICATION_MODEL_PATH)
huyen_classification_model = my_model((224,224,1),2)
huyen_classification_model.load_weights(HUYEN_CLASSIFICATION_MODEL_PATH)

# Vote
vote_map = {0: 0, 1: 0}

if __name__ == '__main__':
    # text detection
    full_cmnd_img = cv2.imread(FULL_CMND_IMG_PATH)
    full_cmnd_img, text_images, datas = detect_text(full_cmnd_img, text_detection_model)
    if IS_SHOW_FULL_IMAGE:
        cv2.imshow('full img', full_cmnd_img)
        cv2.waitKey(0)

    # sign detection
    total_sac_imgs = []
    total_huyen_imgs = []
    for text_image in text_images:
        text_image, sac_crop_imgs, huyen_crop_imgs, datas = detect_sac_huyen(text_image, sac_huyen_detection_model)
        if IS_SHOW_TEXT:
            cv2.imshow('text', text_image)
            cv2.waitKey(0)
        total_sac_imgs += sac_crop_imgs
        total_huyen_imgs += huyen_crop_imgs

    print(f'TOTAL SAC SIGNS = {len(total_sac_imgs)}')
    print(f'TOTAL HUYEN SIGNS = {len(total_huyen_imgs)}')

    # show sac sign
    
    for sac_img in total_sac_imgs:
        sac_img = resize(sac_img, 1000)
        sac_img, c = sign_classifier(sac_img, sac_classification_model)
        vote_map[int(c)] += 1
        if IS_SHOW_SAC_SIGN:
            cv2.imshow('sac', sac_img)
            cv2.waitKey(0)
    
    # show huyen sign
    for huyen_img in total_huyen_imgs:
        huyen_img = resize(huyen_img, 1000)
        huyen_img, c = sign_classifier(huyen_img, huyen_classification_model)
        vote_map[int(c)] += 1
        if IS_SHOW_HUYEN_SIGN:
            cv2.imshow('huyen', huyen_img)
            cv2.waitKey(0)

    # result 
    print(vote_map)
    vote_result = vote_map[0] - vote_map[1]
    if vote_result > 0:
        print('-------REAL ID CARD-------')
    elif vote_result == 0:
        print('-------UNKNOW ID CARD-------')
    else:
        print('-------FAKE ID CARD-------')


    