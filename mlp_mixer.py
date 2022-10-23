import os, glob
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import PIL.Image as Image
import einops

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

from tensorflow.keras.activations import gelu
from tensorflow.keras import layers

tf.keras.backend.set_image_data_format('channels_first')

def mlpBlock(dim:int, name=None):
    def func(x):
        if name == None:
            name1, name2 = None, None
        else:
            name1, name2 = name+'_1', name+'_2'
        y = layers.Dense(units=dim, name=name1)(x)
        y = gelu(y)
        return layers.Dense(x.shape[-1], name=name2)(y)
    return func


mlp_block = mlpBlock

def mixerBlock(token_mlp_dim, channels_mlp_dim, name=None, isNorm=True):
  def func(x):
      if isNorm:
          y = layers.LayerNormalization()(x)
      else:
          y=x
      y = tf.transpose(y, [0, 1, 3, 2])
      y = mlpBlock(token_mlp_dim, name=name+'_token_mixing')(y)
      y = tf.transpose(y, [0, 1, 3, 2])
      x = x + y
      if isNorm:
          y = layers.LayerNormalization()(x)
      else:
          y=x
      y = x + mlpBlock(channels_mlp_dim, name=name+'_channel_mixing')(y)
      return y
  return func

# MLP Mixer model
def mlpMixer(input_shape=(3, 128, 128), patch_size=16, hidden_dim = 512, depth=6, isNorm=True):

    token_mlp_dim, channels_mlp_dim = input_shape[1:]
    print(token_mlp_dim, channels_mlp_dim)

    myinput = layers.Input(shape=input_shape, name='myinput')
    x = layers.Conv2D(filters=hidden_dim, kernel_size=patch_size, strides=patch_size)(myinput)

    for i in range(depth):
        x = mixerBlock(token_mlp_dim=token_mlp_dim, channels_mlp_dim=channels_mlp_dim, name='mlp_block_%d'%i, isNorm=isNorm)(x)

    model = tf.keras.models.Model(myinput, x)
    return model

model = mlpMixer( depth=6, isNorm=False)
model.summary()
