
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
                 input_shape=(64, 64, 4),
                 ch=(16, 32, 64, 128),
                 use_bn=True,
                 add_noise_model_input=False,
                 add_noise_dec_input=False
                 ):


        self.input_shape = input_shape
        self.ch = ch
        self.use_bn = use_bn
        self.add_noise_model_input = add_noise_model_input
        self.add_noise_dec_input = add_noise_dec_input

        self.model_ae = self.ae()
        self.model_aet = self.aet()
        # self.model_ae_delta = self.ae_delta_bw()
        # self.model_vae_delta = self.vae_delta_bw()
        self.model_resnet_ed = self.resnet_ed()
        self.model_resnet_flat = self.resnet_flat()
        self.model_unet = self.unet_generator()
        self.model_bwunet = self.bwunet()

        if model_name == 'bwunet':
            self.model = self.bwunet()
        elif  model_name == 'unet':
            self.model = self.unet_generator()
        elif model_name == 'resnet_ed':
            self.model = self.resnet_ed()
        elif model_name == 'resnet_flat':
            self.model = self.resnet_flat()
        elif model_name == 'ae':
            self.model = self.ae()
        elif model_name == 'aet':
            self.model = self.aet()
        else:
            self.model = None



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
        if type.lower() == 'batch':
            out = tf.keras.layers.BatchNormalization(name=name)
        elif type.lower()== 'instance':
            out = tfa.layers.InstanceNormalization(name=name)
        elif type.lower() == None:
            out = tf.keras.activations.linear(name=name)
        else:
            ValueError('Undefined normalization type, ', type)
        return out

    def downsample(self, filters, size, norm='batch', apply_norm=True, activation='leakyrelu'):

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=True))

        if apply_norm:
            result.add(self._mynorm_layer(norm))

        result.add(self._myactivation_layer(activation))

        return result


    def upsample(self, filters, size, norm='batch', apply_dropout=False, activation='relu'):


        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=True))

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
        last = tf.keras.layers.Conv2DTranspose(
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


    def _enc_block(self, filters, kernek_size, strides:int=1, padding:str='same', pooling='max', name=None):
        def func(x):
            if pooling == None:
                enc = tf.keras.layers.Conv2D(filters, kernek_size, strides,
                                             padding, activation=None, name=name)(x)
                enca = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelu')(enc)
                encp = enca
            else:
                enc = tf.keras.layers.Conv2D(filters, kernek_size, strides,
                                             padding, activation=None, name=name)(x)
                enca = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelu')(enc)
                if pooling == 'max':
                    encp = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name=name+f'_{pooling}pooling')(enca)
                elif pooling == 'avgerage':
                    encp = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name=name+f'_{pooling}pooling')  (enca)
            return enc, enca, encp
        return func

    def _dec_block(self, filters, kernel_size, strides:int=2, padding:str='same', name=None):
        def func(x, res=None):
            out = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding=padding, activation=None, name=name+'_tconv')(x)

            if res != None:
                out = tf.keras.layers.Add()([out, res])
                out = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelua')(out)

            out = tf.keras.layers.Conv2D(filters=filters//2,  kernel_size=kernel_size, strides=1,
                                         padding=padding, activation=None, name=name+'_conv')(out)

            out = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+'_prelu')(out)
            return out
        return func

    def bwunet(self, input_shape=(128, 128, 3)):


        kernel_size = 3
        filters=64

        input = tf.keras.layers.Input(shape=input_shape, name='bwunet_input')

        # emb = tf.keras.layers.Conv2D(filters=filters, kernek_size=kernel_size, strides=1, padding='same',
        #                              activation=None, name='emb')(input)
        emb, emba, embp = self._enc_block(filters, kernel_size, strides=1, pooling=None, name='emb')(input)   # (128, 128, 64)

        # encoder
        enc0, enc0a, enc0p = self._enc_block(filters*2, kernel_size, name='enc0')(embp)       # (64, 64, 128)
        enc1, enc1a, enc1p = self._enc_block(filters*4, kernel_size, name='enc1')(enc0p)      # (32, 32, 256)
        enc2, enc2a, enc2p = self._enc_block(filters*8, kernel_size, name='enc2')(enc1p)      # (16, 16, 512)
        enc3, enc3a, enc3p = self._enc_block(filters*16, kernel_size, name='enc3')(enc2p)     # ( 8,  8, 1024)

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

        out = tf.keras.layers.Conv2D(filters=3, kernel_size=kernel_size, strides=1,
                                    padding='same', activation='tanh', name='last0')(last1p)

        model = tf.keras.Model(inputs=input, outputs=out, name='bwunet')
        return model



    def _residual_block(self, x, nch:int, ctype:str, nblock:int=0, norm='batch', ):
        normlist = [None, 'batch', 'instance']
        assert norm in normlist, f'form should be in {normlist}, but {norm}'

        out = tf.keras.layers.Conv2D(filters=nch, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation=None, name=f'{ctype}_conv_{nblock}_1')(x)
        out = self._mynorm(out, norm, nblock, 1)
        out = tf.keras.layers.ReLU(name=f'enc_relu_{nblock}_1')(out)

        out = tf.keras.layers.Conv2D(filters=nch, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation=None, name=f'{ctype}_conv_{nblock}_2')(out)
        out = self._mynorm(out, norm, nblock, 2)
        out = tf.keras.layers.Add()([x, out])
        out = tf.keras.layers.ReLU(name=f'{ctype}_relu_{nblock}_2')(out)

        return out

    def resnet_flat(self, input_shape=(128, 128, 3), nch:int=64, nblocks:int=12, norm='batch'):
        input = tf.keras.layers.Input(shape=input_shape, name='resnet_flat_input')


        out = tf.keras.layers.Conv2D(filters=nch,
                                    kernel_size=(3, 3),
                                    strides=(1,1),
                                    padding='same',
                                    activation='relu',
                                    name='enc0')(input)


        for idx in range(nblocks):
            out = self._residual_block(out, nch=nch, ctype='flat', nblock=idx, norm=norm)



        out = tf.keras.layers.Conv2D(filters=3,
                                    kernel_size=(3, 3),
                                    strides=(1,1),
                                    padding='same',
                                    activation='tanh',
                                    name='last')(out)

        model = tf.keras.Model(inputs=input, outputs=out, name=f'resnet_flat_{nblocks}')
        return model



    def resnet_ed(self, input_shape=(128, 128, 3), nch=64, num_downsampling_blocks=2,num_residual_blocks=9,num_upsample_blocks=2):

        # gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        norm = 'batch'
        input = tf.keras.layers.Input(shape=input_shape, name='resnet_ed_input')



        out = tf.keras.layers.Conv2D(filters=nch,
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
        out = tf.keras.layers.Conv2D(filters=3, kernel_size=(7, 7), strides=(1,1),
                                    padding='same', activation='tanh', name='final')(out)
        model = tf.keras.models.Model(input, out, name=f'resnet_ed_{num_residual_blocks}')
        return model


    def ae(self, input_shape=(64, 64, 4)):
        ch = self.ch

        input = tf.keras.layers.Input(shape=input_shape, name='input')


        ## encoder
        # enc1
        x1 = tf.keras.layers.Conv2D(filters=ch[0], kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation='relu', name='enc_conv1')(input)
        # enc2
        x2 = tf.keras.layers.Conv2D(filters=ch[1], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv2')(x1)
        # enc3
        x3 = tf.keras.layers.Conv2D(filters=ch[2], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv3')(x2)

        # enc4
        x4 = tf.keras.layers.Conv2D(filters=ch[3], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv4')(x3)


        if self.use_bn == True:
            print('encoder output is batch normalized!!!')
            x4 = tf.keras.layers.BatchNormalization(name='enc_bn')(x4)




        ## decoder
        # dec3
        x = tf.keras.layers.Conv2DTranspose(filters=ch[2], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation='relu', name='dec_tconv3')(x4)
        x = tf.keras.layers.Conv2D(filters=ch[2], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv3_conv1')(x)

        # dec2
        x = tf.keras.layers.Conv2DTranspose(filters=ch[1], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation='relu', name='dec_tconv2')(x)

        x = tf.keras.layers.Conv2D(filters=ch[1], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv2_conv1')(x)

        # dec1
        x = tf.keras.layers.Conv2DTranspose(filters=ch[0], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation=None, name='dec_tconv1')(x)

        x = tf.keras.layers.Add()([x, x1 ])

        x = tf.keras.layers.Conv2D(filters=ch[0], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv1_conv1')(x)

        # last
        x = tf.keras.layers.Conv2DTranspose(filters=ch[0], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation=None, name='dec_tconv_last')(x)
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, name='dec_conv_last')(x)

        model = tf.keras.models.Model(input, x, name='ae')

        return model


    def aet(self, input_shape=(128, 128, 3)):
        ch = self.ch

        input = tf.keras.layers.Input(shape=input_shape, name='input')


        ## encoder
        # enc1
        x1 = tf.keras.layers.Conv2D(filters=ch[0], kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation='relu', name='enc_conv1')(input)
        # enc2
        x2 = tf.keras.layers.Conv2D(filters=ch[1], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv2')(x1)
        # enc3
        x3 = tf.keras.layers.Conv2D(filters=ch[2], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv3')(x2)

        # enc4
        x4 = tf.keras.layers.Conv2D(filters=ch[3], kernel_size=(3, 3), strides=(2, 2),
                                    padding='same', activation='relu', name='enc_conv4')(x3)


        if self.use_bn == True:
            print('encoder output is batch normalized!!!')
            x4 = tf.keras.layers.BatchNormalization(name='enc_bn')(x4)




        ## decoder
        # dec3
        x = tf.keras.layers.Conv2DTranspose(filters=ch[2], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation='relu', name='dec_tconv3')(x4)
        x = tf.keras.layers.Conv2D(filters=ch[2], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv3_conv1')(x)

        # dec2
        x = tf.keras.layers.Conv2DTranspose(filters=ch[1], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation='relu', name='dec_tconv2')(x)

        x = tf.keras.layers.Conv2D(filters=ch[1], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv2_conv1')(x)

        # dec1
        x = tf.keras.layers.Conv2DTranspose(filters=ch[0], kernel_size=(3, 3), strides=(2, 2),
                                            bias_initializer='glorot_uniform',
                                            padding='same', activation=None, name='dec_tconv1')(x)

        x = tf.keras.layers.Add()([x, x1 ])

        x = tf.keras.layers.Conv2D(filters=ch[0], kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation='relu', name='dec_tconv1_conv1')(x)

        # last
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, name='dec_conv_last')(x)

        model = tf.keras.models.Model(input, x, name='aet')

        return model








def main():



    model_name = 'resnet_ed'
    # model_name = 'resnet_flat'
    # model_name = 'bwunet'
    # model_net = 'unet'

    bw = GenerationTF(model_name =  model_name)
    save_as_tflite(bw.model, name=f'model_{model_name}' )



if __name__ == '__main__':
    main()