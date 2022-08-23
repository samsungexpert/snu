
from multiprocessing.sharedctypes import Value
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import numpy as np

import os

TF_VER=1 if tf.__version__.split('.')[0]=='1' else (2 if tf.__version__.split('.')[0]=='2' else None)


def save_as_tflite(model, name='model'):
    model.input.set_shape(1 + model.input.shape[1:])
    model.save(name + '.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(name + '.tflite', 'wb').write(tflite_model)

    # to json
    model_json = model.to_json()
    open(name + '.json', 'w').write(model_json)





class GenerationTF():

    def __init__(self,
                 model_name='bwunet',
                 input_shape=(128, 128, 3),
                 ch=(16, 32, 64, 128),
                 use_bn=True,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 ):


        self.input_shape = input_shape
        self.ch = ch

        self.use_bn = use_bn

        self.kernel_regularizer = None
        if kernel_regularizer != None:
            self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.01)

        self.kernel_constraint = None
        if kernel_constraint != None:
            self.kernel_constraint = tf.keras.constraints.MinMaxNorm(
                    min_value=0.0, max_value=kernel_constraint, rate=1.0, axis=[0, 1, 2] )
            self.bias_constraint = tf.keras.constraints.MinMaxNorm(
                    min_value=0.0, max_value=kernel_constraint, rate=1, axis=0)

        if model_name == 'bwunet':
            self.model = self.bwunet(self.input_shape)
        elif model_name == 'unet':
            self.model = self.unet_generator(self.input_shape)
        elif model_name == 'resnet_ed':
            self.model = self.resnet_ed(self.input_shape)
        elif model_name == 'resnet_flat':
            self.model = self.resnet_flat(self.input_shape)
        else:
            print('====================== unknown model name, ', model_name)
            exit()

        print(f'-----------> {model_name}, init done <-------------')


    def _myactivation_layer(self, activation='relu'):
        if activation.lower() == 'relu':
            activation_layer = tf.keras.layers.ReLU()
        elif activation.lower() == 'leakyrelu':
            activation_layer = tf.keras.layers.LeakyReLU()
        else:
            activation_layer = tf.keras.activations.linear()
        return activation_layer


    def _mynorm(self, x, type, nblock, num):
        out = self._mynorm_layer(type)(x)
        return out

    def _mynorm_layer(self, type, name=None):
        if not isinstance(type, str):
            out = tf.keras.activations.linear
        elif type.lower() == 'batch':
            out = tf.keras.layers.BatchNormalization(name=name)
        elif type.lower()== 'instance':
            out = tfa.layers.InstanceNormalization(name=name)
        elif type.lower() == 'None':
            out = tf.keras.activations.linear
        else:
            ValueError('Undefined normalization type, ', type)
        return out


    def conv2d(self, filters, kernel_size, strides=1, padding='same',
                kernel_initializer='glorot_uniform', bias_initializer='zeros', use_bias=True,  activation=None, name=None):
        return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer, bias_regularizer=None,
                                    kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint,
                                    padding=padding, activation=activation, name=name)

    def tconv2d(self, filters, kernel_size, strides=2, padding='same',
                kernel_initializer='glorot_uniform', bias_initializer='zeros', use_bias=True,  activation=None, name=None):
        return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                            kernel_regularizer=self.kernel_regularizer, bias_regularizer=None,
                                            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint,
                                            padding=padding, activation=activation, name=name)

    def downsample(self, filters, size, norm='batch', apply_norm=True, activation='leakyrelu'):

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            self.conv2d(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=True))

        if apply_norm:
            result.add(self._mynorm_layer(norm))

        result.add(self._myactivation_layer(activation))

        return result


    def upsample(self, filters, size, norm='batch', apply_dropout=False, activation='relu'):


        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(self.tconv2d(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer,use_bias=True))

        result.add(self._mynorm_layer(norm.lower()))

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        if activation.lower() == 'relu':
            activation_layer = tf.keras.layers.ReLU()
        elif activation.lower() == 'leakyrelu':
            activation_layer = tf.keras.layers.LeakyReLU()
        else:
            activation_layer = tf.keras.activations.linear()

        result.add(activation_layer)

        return result


    def unet_generator(self, input_shape=(128, 128, 3), output_channels=3, norm='batch'):



        down_stack = [
            self.downsample(64, 4, norm, apply_norm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4, norm),  # (bs, 64, 64, 128)
            self.downsample(256, 4, norm),  # (bs, 32, 32, 256)
            self.downsample(512, 4, norm),  # (bs, 16, 16, 512)
            self.downsample(512, 4, norm),  # (bs, 8, 8, 512)
            self.downsample(512, 4, norm),  # (bs, 4, 4, 512)
            self.downsample(512, 4, norm),  # (bs, 2, 2, 512)
            # self.downsample(512, 4, norm),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            # self.upsample(512, 4, norm, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, norm, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, norm, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4, norm),  # (bs, 16, 16, 1024)
            self.upsample(256, 4, norm),  # (bs, 32, 32, 512)
            self.upsample(128, 4, norm),  # (bs, 64, 64, 256)
            self.upsample(64, 4, norm),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        last = self.tconv2d(
            output_channels, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=input_shape, name='unet_input')
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            # x = concat([x, skip])
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name='unet')


    def _enc_block(self, filters:int, kernek_size:int, strides:int=1, padding:str='same', pooling:str='max', norm:str='batch', name=None):
        def func(x):
            enc = self.conv2d(filters, kernek_size, strides,
                                             padding, activation=None, name=name)(x)
            enc = self._mynorm_layer(norm)(enc)
            enca = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelu')(enc)

            if pooling == 'max':
                encp = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name=name+f'_{pooling}pooling')(enca)
            elif pooling == 'avgerage':
                encp = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name=name+f'_{pooling}pooling')  (enca)
            else:
                encp = enca
            return enc, enca, encp
        return func

    def _dec_block(self, filters:int, kernel_size:int, strides:int=2, padding:str='same', norm:str='batch', name=None):
        def func(x, res=None):
            out = self.tconv2d(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding=padding, activation=None, name=name+'_tconv')(x)
            out = self._mynorm_layer(norm)(out)
            if res != None:
                out = tf.keras.layers.Add()([out, res])
                out = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelua')(out)

            out = self.conv2d(filters=filters//2,  kernel_size=kernel_size, strides=1,
                                         padding=padding, activation=None, name=name+'_conv')(out)
            out = self._mynorm_layer(norm)(out)
            out = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelu')(out)
            return out
        return func

    def bwunet(self, input_shape=(128, 128, 3), norm='batch'):


        kernel_size = 3
        filters=64

        input = tf.keras.layers.Input(shape=input_shape, name='bwunet_input')

        emb, emba, embp = self._enc_block(filters, kernel_size, strides=1, pooling=None, norm=None, name='emb')(input)   # (128, 128, 64)

        # encoder
        enc0, enc0a, enc0p = self._enc_block(filters*2,  kernel_size, norm='batch', name='enc0')(embp)       # (64, 64, 128)
        enc1, enc1a, enc1p = self._enc_block(filters*4,  kernel_size, norm='batch', name='enc1')(enc0p)      # (32, 32, 256)
        enc2, enc2a, enc2p = self._enc_block(filters*8,  kernel_size, norm='batch', name='enc2')(enc1p)      # (16, 16, 512)
        enc3, enc3a, enc3p = self._enc_block(filters*16, kernel_size, norm='batch', name='enc3')(enc2p)     # ( 8,  8, 1024)

        # flat

        _, _, flat0p = self._enc_block(filters*16, kernel_size, pooling=None, name='flat0')(enc3p)  # ( 8,  8, 1024)
        _, _, flat1p = self._enc_block(filters*16, kernel_size, pooling=None, name='flat1')(flat0p) # ( 8,  8, 1024)


        # decoder
        dec2 = self._dec_block(filters*16, kernel_size, name='dec3')(flat1p, enc3) # (  8,   8, 1024 ) -> (  16,  16, 512)
        dec1 = self._dec_block(filters*8 , kernel_size, name='dec2')(dec2, enc2)   # ( 16,  16,  512 ) -> (  32,  32, 256)
        dec0 = self._dec_block(filters*4,  kernel_size, name='dec1')(dec1, enc1)   # ( 32,  32,  256)  -> (  64,  64, 128)
        dec  = self._dec_block(filters*2,  kernel_size, name='dec0')(dec0, enc0)   # ( 64,  64,  128)  -> ( 128, 128,  64)

        # last
        last1, last1a, last1p = self._enc_block(filters//2, kernel_size, pooling=None, name='last1')(dec)

        out = self.conv2d(filters=3, kernel_size=kernel_size, strides=1,
                                    padding='same', activation='tanh', name='last0')(last1p)

        model = tf.keras.Model(inputs=input, outputs=out, name='bwunet')
        return model



    def _residual_block(self, x, nch:int, ctype:str, nblock:int=0, norm='batch', ):
        normlist = [None, 'batch', 'instance']
        assert norm in normlist, f'form should be in {normlist}, but {norm}'

        out = self.conv2d(filters=nch, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation=None, name=f'{ctype}_conv_{nblock}_1')(x)
        out = self._mynorm(out, norm, nblock, 1)
        out = tf.keras.layers.ReLU(name=f'enc_relu_{nblock}_1')(out)

        out = self.conv2d(filters=nch, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation=None, name=f'{ctype}_conv_{nblock}_2')(out)
        out = self._mynorm(out, norm, nblock, 2)
        out = tf.keras.layers.Add()([x, out])
        out = tf.keras.layers.ReLU(name=f'{ctype}_relu_{nblock}_2')(out)

        return out

    def resnet_flat(self, input_shape=(128, 128, 3), nch:int=128, nblocks:int=12, norm='batch'):
        input = tf.keras.layers.Input(shape=input_shape, name='resnet_flat_input')


        out = self.conv2d(filters=nch,
                        kernel_size=(3, 3),
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                        name='enc0')(input)

        rout = []
        for idx in range(nblocks):
            out = self._residual_block(out, nch=nch, ctype='flat', nblock=idx, norm=norm)
            rout.append(out)
            if idx>nblocks//2:
                out += rout[nblocks-1-idx]


        out = self.conv2d(filters=nch,
                        kernel_size=(3, 3),
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                        name='last1')(out)

        out = self.conv2d(filters=3,
                        kernel_size=(3, 3),
                        strides=(1,1),
                        padding='same',
                        activation='tanh',
                        name='last0')(out)

        model = tf.keras.Model(inputs=input, outputs=out, name=f'resnet_flat_{nblocks}')
        return model



    def resnet_ed(self, input_shape=(128, 128, 3), nch=64, num_downsampling_blocks=2,num_residual_blocks=9,num_upsample_blocks=2):

        # gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        norm = 'batch'
        input = tf.keras.layers.Input(shape=input_shape, name='resnet_ed_input')

        out = self.conv2d(filters=nch,
                        kernel_size=(7, 7),
                        strides=(1,1),
                        padding='same',
                        activation=None,
                        name='enc0')(input)

        out = self._mynorm(out, type=norm, nblock=0, num=0)
        out = tf.keras.layers.ReLU(name='enc_relu0')(out)


        # Downsampling
        for _ in range(num_downsampling_blocks):
            nch *= 2
            out = self.downsample(filters=nch, size=3, activation='relu')(out)

        # Residual blocks
        for idx in range(num_residual_blocks):
            out = self._residual_block(out, nch=nch, ctype='flat', nblock=idx, norm='batch')

        # Upsampling
        for _ in range(num_upsample_blocks):
            nch //= 2
            out = self.upsample(filters=nch, size=3, activation='relu')(out)

        # Final block
        out = self.conv2d(filters=3, kernel_size=(7, 7), strides=(1,1),
                                    padding='same', activation='tanh', name='final')(out)

        model = tf.keras.models.Model(input, out, name=f'resnet_ed_{num_residual_blocks}')
        return model









def main():


    model_name = []
    model_name.append('bwunet')
    model_name.append('unet')
    model_name.append('resnet_ed')
    model_name.append('resnet_flat')
    for mn in model_name:
        print('model_name = ', mn)
        bw = GenerationTF(model_name= mn, kernel_regularizer=True, kernel_constraint=True)
    # # save_as_tflite(bw.model, name=f'model_{model_name}' )
        bw.model.save(f'{mn}_sw.h5')


    # model_name = 'resnet_ed'
    # model_name = 'resnet_flat'
    # model_name = 'bwunet'
    # model_name = 'unet'
    # bw = GenerationTF(model_name =  model_name, kernel_regularizer=True, kernel_constraint=True)
    # # save_as_tflite(bw.model, name=f'model_{model_name}' )
    # bw.model.save('sw.h5')

    # a = None
    # print(isinstance (a, str))
    # print(type(a))

if __name__ == '__main__':
    main()