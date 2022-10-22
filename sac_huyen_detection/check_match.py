import os
import cv2

IMG_PATH = 'images'
LABEL_PATH = 'labels'
IS_MATCH_CHECK = False

if __name__ == '__main__':
    # Check matching against
    imgs = os.listdir(IMG_PATH)
    labels = os.listdir(LABEL_PATH)
    
    imgs = [i[:-5] for i in imgs]
    labels = [l[:-4] for l in labels]
    print(f'check imgs')
    for i in imgs:
        if i not in labels:
            print(i)
    
    print(f'check labels')
    for l in labels:
        if l not in imgs:
            print(l)