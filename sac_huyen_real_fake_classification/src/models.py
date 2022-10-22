
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization, AveragePooling2D
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU, GlobalAvgPool2D

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf

from time import time
import numpy as np

def my_model(input_shape, n_classes):
  print('ooooooooooooooooooooooo')
  input = Input(input_shape)
  
  x = Conv2D(64, 3, padding='same', activation=tf.nn.relu)(input)
  x = AveragePooling2D(pool_size=2,strides=2)(x)
  
  x = Conv2D(128, 3, padding='same', activation=tf.nn.relu)(x)
  x = AveragePooling2D(pool_size=2,strides=2)(x)
  
  x = Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)
  x = Conv2D(128, 1, padding='same', activation=tf.nn.relu)(x)
  x = Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)
  x = AveragePooling2D(pool_size=2,strides=2)(x)

  x = Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)

  x = Flatten()(x)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

if __name__ == '__main__':
  model_cnn = my_model((224,224,1),2)
  model_cnn.summary()