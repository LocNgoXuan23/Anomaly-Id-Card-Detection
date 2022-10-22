
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
import tensorflow as tf

def my_model(input_shape, n_classes):
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