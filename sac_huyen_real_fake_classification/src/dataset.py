import json
import os
import numpy as np
import tensorflow as tf
import cv2
from utils import *
from tqdm import tqdm

def get_data(path):
	data = read_json(path)

	imgs = []
	labels = []
	paths = []

	for d in tqdm(data):
		label = get_label(d)
		img = cv2.imread(d[:-2])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (224, 224))
		labels.append(label)
		imgs.append(img)
		paths.append(d)
	
	imgs = np.array(imgs)
	labels = np.array(labels)
	paths = np.array(paths)

	return [imgs, labels, paths]


if __name__ == '__main__':
	test_data = get_data('../data/val.json')
	print(test_data[0].shape)
