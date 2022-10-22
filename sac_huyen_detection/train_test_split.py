import os
import cv2
import shutil
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_PATH = 'images'
LABEL_PATH = 'labels'
TRAIN_ROOT = 'train'
VAL_ROOT = 'val'

if __name__ == '__main__':
    # make folder for train and validation
    shutil.rmtree(TRAIN_ROOT)
    shutil.rmtree(VAL_ROOT)
    os.makedirs(TRAIN_ROOT)
    os.makedirs(os.path.join(TRAIN_ROOT, 'images'))
    os.makedirs(os.path.join(TRAIN_ROOT, 'labels'))
    os.makedirs(VAL_ROOT)
    os.makedirs(os.path.join(VAL_ROOT, 'images'))
    os.makedirs(os.path.join(VAL_ROOT, 'labels'))
    
    imgs = os.listdir(IMG_PATH)
    labels = os.listdir(LABEL_PATH)

    imgs.sort()
    labels.sort()

    img_train, img_test, label_train, label_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)

    # print(img_train[50])
    # print(label_train[50])

    # print(img_test[5])
    # print(label_test[5])

    # Train 
    for i in tqdm(img_train):
        tmp = os.path.join(IMG_PATH, i)
        img = cv2.imread(tmp)
        cv2.imwrite(os.path.join(TRAIN_ROOT, 'images', i), img)

    for l in tqdm(label_train):
        shutil.copyfile(os.path.join(LABEL_PATH, l), os.path.join(TRAIN_ROOT, 'labels', l))

    # Test
    for i in tqdm(img_test):
        tmp = os.path.join(IMG_PATH, i)
        img = cv2.imread(tmp)
        cv2.imwrite(os.path.join(VAL_ROOT, 'images', i), img)

    for l in tqdm(label_test):
        shutil.copyfile(os.path.join(LABEL_PATH, l), os.path.join(VAL_ROOT, 'labels', l))


    