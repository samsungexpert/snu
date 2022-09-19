

import os
import random
import glob
import tensorflow as tf
import numpy as np
from functools import partial

from tensorflow.keras.callbacks import Callback
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte

from collections import OrderedDict

TETRA = 2
NONA = 3
SEDEC = 4

TF_VER=1 if tf.__version__.split('.')[0]=='1' else (2 if tf.__version__.split('.')[0]=='2' else None)



def load_checkpoint_if_exists(model, model_dir, model_name, ckpt_name=None):
    prev_epoch = 0

    prev_loss = np.inf
    trained_model_file_name = os.path.join(model_dir, ckpt_name)
    if (ckpt_name != None ) and \
       (os.path.isfile(trained_model_file_name) and os.path.exists(trained_model_file_name)):

        print('=============> load designated trained model: ', ckpt_name)
        model.load_weights(trained_model_file_name)
        idx = trained_weights.rfind(model_name)
        prev_epoch = int(trained_weights[idx-6:idx-1])
        prev_loss = float(trained_weights.split('_')[-1][:-3])

        # raise ValueError('unkown trained weight', trained_model_file_name)
    else:
        if (trained_model_file_name != None ) and (not os.path.exists(trained_model_file_name)):
            print('=============> pre trained designated model not exists ')

        trained_weights = glob.glob(os.path.join(model_dir, '*.h5' ))
        print('--=-=-=-> ', trained_weights)
        if len(trained_weights ) > 0:
            print('===========> %d TRAINED WEIGHTS EXIST' % len(trained_weights))
            trained_weights.sort()
            trained_weights = trained_weights[-1]
            print('---------------------> ', trained_weights)
            model.load_weights(trained_weights)
            idx = trained_weights.rfind(model_name)
            prev_epoch = int(trained_weights[idx-6:idx-1])
            prev_loss = float(trained_weights.split('_')[-1][:-3])
            print('prev epoch', prev_epoch)
        else:
            print('===========> TRAINED WEIGHTS NOT EXIST', len(trained_weights))

    return model, prev_epoch, prev_loss




class bwutils():

    def __init__(self,
                input_type='rgb',
                output_type='data_only', # 'data_only', 'data_with_mask'
                cfa_pattern='tetra',
                patch_size:int=128,
                crop_size:int=128,
                input_max = 1.,
                input_bits =16,
                input_bias = True,
                use_unprocess=False,
                loss_scale = 1,
                alpha_for_gamma = 0.05,
                beta_for_gamma = (1./2.2),
                upscaling_factor=None,
                upscaling_method='bilinear',
                loss_type=['rgb', 'ploss'], # 'rgb', 'yuv', 'ploss', 'ssim'.
                loss_mode='2norm',
                cache_enable=False):

        if input_type not in ['shrink', 'nonshrink', 'nonshrink_4ch', 'shrink_upscale', 'raw_1ch', 'rgb']:
            raise ValueError('unknown input_type  '
                             'input type must be either "shrink" / "nonshrink" / "nonshrink_4ch" / "raw_1ch"  but', input_type)

        cfa_pattern = cfa_pattern.lower()
        if cfa_pattern in ['bayer', 1]:
            cfa_pattern = 1
        elif cfa_pattern in ['tetra', 2]:
            cfa_pattern = 2
        elif cfa_pattern in ['nona', 3]:
            cfa_pattern = 3
        elif cfa_pattern in ['sedec', 4]:
            cfa_pattern = 4
        else:
            raise ValueError('unknown cfa_pattern, ', cfa_pattern)


        for lt  in loss_type:
            if lt not in ['rgb', 'yuv', 'ploss', 'ssim']:
                raise ValueError('unknown loss type, ', lt)

        self.cfa_pattern = cfa_pattern
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.input_bits = input_bits
        self.input_max = input_max
        self.input_bias = input_bias
        self.use_unprocess = use_unprocess
        self.alpha_for_gamma = alpha_for_gamma
        self.beta_for_gamma =  beta_for_gamma
        self.upscaling_factor = upscaling_factor
        self.upscaling_method = upscaling_method
        self.loss_type = loss_type
        self.loss_scale = loss_scale
        self.cache_enable = cache_enable

        self.input_type = input_type
        self.output_type = output_type

        # self.input_scale = input_max / 255.

        if loss_mode == 'square' or loss_mode == 'mse' or loss_mode=='2norm':
            self.loss_norm = tf.keras.backend.square
        elif loss_mode == 'abs' or loss_mode=='1norm':
            self.loss_norm = tf.keras.backend.abs
        else:
            ValueError('unknown loss_mode %s' %  loss_mode)


        self.crop_margin = int(patch_size - crop_size)


        self.idx_R = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G1 = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G2 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_B = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))





        self.idx_RGB = np.concatenate((self.idx_R[..., np.newaxis],
                                       self.idx_G[..., np.newaxis],
                                       self.idx_B[..., np.newaxis]), axis=-1)

        self.idx_G1RBG2 = np.concatenate((self.idx_G1[..., np.newaxis],
                                          self.idx_R[..., np.newaxis],
                                          self.idx_B[..., np.newaxis],
                                          self.idx_G2[..., np.newaxis]), axis=-1)


        if 'ploss' in loss_type:
            vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                    input_shape=[crop_size,crop_size,3])
            loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
            loss_model.trainable = False
            self.loss_model = [loss_model]

            # scaling_factor =  1. / (12.75 * 12.75) #5.4 # 2.2 wo gan
            # scaling_factor = 2. / 162.5625  # 5.4 # 2.2 wo gan
            #self.ploss_scaling_factor = 2. / 162.5625
            self.ploss_scaling_factor = 4. / 162.5625 # in SRGAN  MSE RGB (-1~1), ploss 1/12.75
            if loss_mode == '1norm':
                self.ploss_scaling_factor = np.sqrt(4. / 162.5625) # tf.keras.backend.sqrt(self.ploss_scaling_factor)

        print('[bwutils] input_type', input_type)
        print('[bwutils] output_type', output_type)
        print('[bwutils] cfa_pattern', cfa_pattern)
        print('[bwutils] patch_size', patch_size)
        print('[bwutils] crop_size', crop_size)
        print('[bwutils] upscaling_factor', upscaling_factor)
        print('[bwutils] input_max', input_max)
        print('[bwutils] loss_type', loss_type)
        print('[bwutils] loss_mode', loss_mode, self.loss_norm)
        print('[bwutils] loss_scale', loss_scale)
        print('[bwutils] cache_enable', cache_enable)


    def serving_input_receiver_fn(self):
        """
        This is used to define inputs to serve the model.
        :return: ServingInputReciever
        """
        # key must be same as model input
        reciever_tensors = OrderedDict(
                # The size of input image is flexible.
                [('input', tf.compat.v1.placeholder(tf.float32, [None, self.crop_size, self.crop_size, 3]))]
        )

        # Convert give inputs to adjust to the model.
        features = reciever_tensors
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                        features=features)


    def save_models(self, model, path, order):
        # init_variables()


        # to h5
        model.save(path + '_%s.h5' % order, include_optimizer=False)

        # model.input.set_shape(1 + model.input.shape[1:]) # to freeze model

        # to tflite
        # if tf.__version__.split('.')[0] == '1':
        #     print('tf version 1.x')
        #     converter = tf.lite.TFLiteConverter.from_keras_model_file(path + '_%s.h5' % order)
        #     tflite_model = converter.convert()
        #     open(path + '_%s.tflite' % order, "wb").write(tflite_model)
        # elif tf.__version__.split('.')[0] == '2':
        #     print('tf version 2.x')
        #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # tflite_model = converter.convert()
        # open(path + '_%s.tflite' % order, "wb").write(tflite_model)

        # # to yaml
        # model_string = model.to_yaml()
        # open(path + '.yaml', 'w').write(model_string)

        # to json
        model_json = model.to_json()
        open(path + '.json', 'w').write(model_json)


    def data_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, np.random.randint(4))
        return image


    def get_image_from_single_example(self, example, key='image', num_channels=3, dtype='uint16'):

        patch_size = self.patch_size

        feature = {
            key: tf.io.FixedLenFeature((), tf.string)
        }
        parsed = tf.io.parse_single_example(example, feature)

        image = tf.io.decode_raw(parsed[key], dtype)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [patch_size, patch_size, num_channels])

        return image


    def get_patternized_1ch_raw_image(self, image):
        patternized = self.get_patternized_3ch_image(image)
        patternized = tf.expand_dims(tf.reduce_sum(patternized, axis=-1), axis=-1)
        return patternized


    def get_patternized_3ch_image(self, image):
        tf_RGB = tf.constant(self.idx_RGB, dtype=tf.float32)
        patternized = tf.math.multiply(image[...,:3], tf_RGB)
        return patternized


    def get_patternized_4ch_image(self, image, is_shrink=True):

        if is_shrink:
            dtype = tf.bool
        else:
            dtype = tf.float32

        tf_G1 = tf.constant(self.idx_G1, dtype=dtype)
        tf_R  = tf.constant(self.idx_R,  dtype=dtype)
        tf_B  = tf.constant(self.idx_B,  dtype=dtype)
        tf_G2 = tf.constant(self.idx_G2, dtype=dtype)

        if is_shrink:
            if self.upscaling_factor == None:
                crop_size = self.crop_size
            else:
                crop_size = int( self.crop_size // self.upscaling_factor)
            G1 = tf.reshape(image[:, :, 1][tf_G1], (crop_size // 2, crop_size // 2))
            R  = tf.reshape(image[:, :, 0][tf_R],  (crop_size // 2, crop_size // 2))
            B  = tf.reshape(image[:, :, 2][tf_B],  (crop_size // 2, crop_size // 2))
            G2 = tf.reshape(image[:, :, 1][tf_G2], (crop_size // 2, crop_size // 2))
        else: # non_shrink
            G1 = tf.math.multiply(image[:, :, 1], tf_G1)
            R  = tf.math.multiply(image[:, :, 0], tf_R)
            B  = tf.math.multiply(image[:, :, 2], tf_B)
            G2 = tf.math.multiply(image[:, :, 1], tf_G2)


        pattenrized = tf.concat((tf.expand_dims(G1, axis=-1),
                                 tf.expand_dims(R, axis=-1),
                                 tf.expand_dims(B, axis=-1),
                                 tf.expand_dims(G2, axis=-1)),
                                axis=-1)

        return pattenrized

    def get_patternized_1ch_to_3ch_image(self, image):

        dtype = tf.float32


        tf_R  = tf.constant(self.idx_R,  dtype=dtype)
        tf_G = tf.constant(self.idx_G, dtype=dtype)
        tf_B  = tf.constant(self.idx_B,  dtype=dtype)


        # non_shrink
        R  = tf.math.multiply(image, tf_R)
        G  = tf.math.multiply(image, tf_G)
        B  = tf.math.multiply(image, tf_B)



        pattenrized = tf.concat((tf.expand_dims(R, axis=-1),
                                 tf.expand_dims(G, axis=-1),
                                 tf.expand_dims(B, axis=-1)),
                                axis=-1)

        return pattenrized

    def resize_to_scale_factor(self, image, crop_size_resized):
        # print('crop_size_resized', crop_size_resized)
        image_resized = tf.image.resize(image,
                                        [crop_size_resized, crop_size_resized],
                                        method=self.upscaling_method,
                                        preserve_aspect_ratio=True,
                                        antialias=True)
        return image_resized


    def get_patternized(self, image, input_type):

        if self.crop_size < self.patch_size:
            dim=3
            image = tf.image.random_crop(image, [self.crop_size, self.crop_size, dim])

        if input_type in ['shrink']:

            patternized = self.get_patternized_4ch_image(image, is_shrink=True)

        elif input_type in ['nonshrink_4ch']:

            patternized = self.get_patternized_4ch_image(image, is_shrink=False)

        elif input_type in ['raw_1ch']:

            patternized = self.get_patternized_1ch_raw_image(image)

        elif input_type in ['nonshrink']:

            patternized = self.get_patternized_3ch_image(image)

        else:
            ValueError('unknown input type', input_type)

        return patternized


    # scale_by_input_max
    def scale_by_input_max(self, image):
        image = image / self.input_max
        return image


    def gamma(self, image):
        alpha = self.alpha_for_gamma
        beta = self.beta_for_gamma
        gammas = 1 + np.random.randn(3)*alpha
        gammas[1] = 1
        gammas *= beta
        image = image ** gammas
        return image


    def add_noise_batch(self, image):
        '''
        image ~ (0,1) normalized
        '''
        noise_gaussian = tf.random.normal(image.shape) * self.input_bits
        noise_gaussian = self.scale_by_input_max(noise_gaussian)

        noise_poisson = tf.random.normal(image.shape) * tf.math.sqrt(image) * self.input_bits
        noise_poisson = self.scale_by_input_max(noise_poisson)
        print('------------------------------------------')
        print('------------------------------------------')
        print('------------------------------------------')
        print('------------------------------------------')
        print('noise_poisson.shape', noise_poisson.shape)

        image = image + noise_gaussian + noise_poisson

        image = tf.clip_by_value(image, 0, 1)

        return image


    def parse_tfrecord(self, example, mode):

        image = self.get_image_from_single_example(example, key='image', num_channels=3)
        if mode == tf.estimator.ModeKeys.TRAIN:
            image = self.data_augmentation(image)

        image = self.scale_by_input_max(image)
        image = tf.clip_by_value(image, 0, 1)
        image_gamma = self.gamma(image)

        # noise addition
        image = self.add_noise_batch(image)

        if self.input_bias:
             image       = (image       * 2) - 1
             image_gamma = (image_gamma * 2) - 1

        return image_gamma, image

    def parse_tfrecord_demosaic(self, example, mode):
        if self.input_bits == 8:
            dtype = 'uint8'
        elif self.input_bits == 16:
            dtype = 'uint16'

        image = self.get_image_from_single_example(example, key='image', num_channels=3, dtype=dtype)
        if mode == tf.estimator.ModeKeys.TRAIN:
            image = self.data_augmentation(image)

        image = self.scale_by_input_max(image)
        image = tf.clip_by_value(image, 0, 1)

        if self.input_bias:
             image       = (image       * 2) - 1

        patternized = self.get_patternized(image, self.input_type)

        print('====================== patternized.shape ', patternized.shape)
        print('====================== image.shape ', image.shape)

        return patternized, image


    def parse_tfrecord_remosaic(self, example, mode):
        if self.input_bits == 8:
            dtype = 'uint8'
        elif self.input_bits == 16:
            dtype = 'uint16'

        image = self.get_image_from_single_example(example, key='image', num_channels=3, dtype=dtype)
        if mode == tf.estimator.ModeKeys.TRAIN:
            image = self.data_augmentation(image)

        image = self.scale_by_input_max(image)
        image = tf.clip_by_value(image, 0, 1)

        if self.input_bias:
             image       = (image       * 2) - 1

        patternized = self.get_patternized(image, self.input_type)

        print('====================== patternized.shape ', patternized.shape)
        print('====================== image.shape ', image.shape)

        return patternized, image


    def data_augmentation_cycle(self, srgb, raw):

        # flip
        if random.random() > 0.5:
            srgb = tf.image.flip_left_right(srgb)
            raw  = tf.image.flip_left_right(raw)

        if random.random() > 0.5:
            srgb = tf.image.flip_up_down (srgb)
            raw  = tf.image.flip_up_down (raw)

        # rotation
        r = np.random.randint(4)
        srgb = tf.image.rot90(srgb, r)
        raw  = tf.image.rot90(raw, r)

        return srgb, raw


    def parse_tfrecord_cycle(self, example, mode):

        patch_size = self.patch_size

        ## get single image
        feature = {
            'raw':  tf.io.FixedLenFeature([], tf.string),
            'srgb': tf.io.FixedLenFeature([], tf.string)
        }

        parsed = tf.io.parse_single_example(example, feature)


        srgb = tf.io.decode_raw(parsed['srgb'], 'uint8')
        raw  = tf.io.decode_raw(parsed['raw'], 'uint16')

        # reshape
        srgb = tf.reshape(srgb, (patch_size, patch_size, 3))
        raw  = tf.reshape(raw,  (patch_size, patch_size))

        # cast & normalize
        srgb = tf.cast(srgb, tf.float32) / (2**8 - 1)
        raw  = tf.cast(raw,  tf.float32) / (2**16 -1)

        # raw 1ch to 3ch
        print('>>>>>>> raw.shape', raw.shape)
        raw = self.get_patternized_1ch_to_3ch_image(raw)
        print('<<<<<<< raw.shape', raw.shape)

        # unprocess
        def unprocess(x, m):
            # de-ltm
            x = 0.5 -tf.math.sin(tf.math.asin(1-2*x)/3) # 3*(x**2) -2*(x**3),  0.5 -np.sin(np.arcsin(1-2*x)/3)

            # de-gamma
            x = tf.math.pow(x, 2.2) # x**(1/2.2)

            # iwb
            if m == tf.estimator.ModeKeys.TRAIN:
                rgain = tf.random.uniform(shape=[1], minval=1.9, maxval=2.4)
                ggain = tf.constant([1.], dtype=tf.float32)
                bgain = tf.random.uniform(shape=[1], minval=1.5, maxval=1.9)
                gain = tf.concat([rgain, ggain, bgain], axis=-1)
            else:
                gain = tf.constant([(1.9+2.4)/2, 1., (1.5+1.9)/2 ], dtype=tf.float32)

            x = x * gain

            return x
        if self.use_unprocess:
            print("USE UNPROCESS, ", self.use_unprocess)
            srgb = unprocess(srgb, mode)



        # augmentation
        if mode == tf.estimator.ModeKeys.TRAIN:
            srgb, raw = self.data_augmentation_cycle(srgb, raw)


        if self.input_bias:
             srgb = (srgb * 2) - 1
             raw  = (raw  * 2) - 1

        return srgb, raw


    # unprocess
    def unprocess(self, image, mode):
        # de-ltm
        image = 0.5 -tf.math.sin(tf.math.asin(1-2*image)/3) # 3*(x**2) -2*(x**3),  0.5 -np.sin(np.arcsin(1-2*x)/3)

        # de-gamma
        image = tf.math.pow(image, 2.2) # x**(1/2.2)

        # add noise
        image = self.add_noise_batch(image)

        # iwb
        if mode == tf.estimator.ModeKeys.TRAIN:
            rgain = tf.random.uniform(shape=[1], minval=1.9, maxval=2.4)
            ggain = tf.constant([1.], dtype=tf.float32)
            bgain = tf.random.uniform(shape=[1], minval=1.5, maxval=1.9)
            gain = tf.concat([rgain, ggain, bgain], axis=-1)
        else:
            gain = tf.constant([(1.9+2.4)/2, 1., (1.5+1.9)/2 ], dtype=tf.float32)

        image = image * gain

        image = tf.clip_by_value(image, 0, 1)

        return image

    def parse_tfrecord_unp(self, example, mode):

        srgb = self.get_image_from_single_example(example, key='image', num_channels=3)
        if mode == tf.estimator.ModeKeys.TRAIN:
            srgb = self.data_augmentation(srgb)

        srgb = self.scale_by_input_max(srgb)

        if self.use_unprocess:
            print("USE UNPROCESS, ", self.use_unprocess)
            raw = self.unprocess(srgb, mode)


        srgb = tf.clip_by_value(srgb, 0, 1)
        raw  = tf.clip_by_value(raw , 0, 1)

        if self.input_bias:
             srgb = (srgb * 2) - 1
             raw  = (raw  * 2) - 1

        return srgb, raw





    def dataset_input_fn(self, params):
        dataset = tf.data.TFRecordDataset(params['filenames'])

        if params['train_type'] == 'unprocessing':
            parse_fn = self.parse_tfrecord_unp
        elif params['train_type'] == 'demosaic':
            parse_fn = self.parse_tfrecord_demosaic

        dataset = dataset.map(partial(parse_fn, mode=params['mode']), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Dataset cache need 120G Main memory
        if self.cache_enable is True:
            dataset = dataset.cache()

        if params['mode'] == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(params['shuffle_buff']).repeat()

        dataset = dataset.batch(params['batch'])

        # dataset = dataset.map(self.add_noise_batch)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.prefetch(8 * params['batch'])

        return dataset

    def dataset_input_fn_cycle(self, params):
        dataset = tf.data.TFRecordDataset(params['filenames'])

        dataset = dataset.map(partial(self.parse_tfrecord_cycle,
                                       mode=params['mode']),
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Dataset cache need 120G Main memory
        if self.cache_enable is True:
            dataset = dataset.cache()

        if params['mode'] == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(params['shuffle_buff']).repeat()

        dataset = dataset.batch(params['batch'])

        # dataset = dataset.map(self.add_noise_batch)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.prefetch(8 * params['batch'])

        return dataset

    def loss_fn(self, y_true, y_pred):

        loss = 0

        if ('rgb' in self.loss_type) and ('yuv' in self.loss_type):
            loss += self.loss_fn_mse_rgb_yuv(y_true, y_pred)
        elif 'rgb' in self.loss_type:
            loss += self.loss_fn_mse_rgb(y_true, y_pred)
        elif 'yuv' in self.loss_type:
            loss += self.loss_fn_mse_yuv(y_true, y_pred)


        if 'ploss' in self.loss_type:
            loss += self.loss_fn_ploss(y_true, y_pred)
        if 'ssim' in self.loss_type:
            loss += self.loss_fn_ssim(y_true, y_pred)

        return loss * self.loss_scale


    def loss_fn_mse_rgb(self, y_true, y_pred):
        rgb_mse_loss = tf.keras.backend.mean(self.loss_norm(y_true - y_pred))
        return rgb_mse_loss


    def loss_fn_mse_yuv(self, y_true, y_pred):
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)

        yuv_mse_loss = tf.keras.backend.mean(
                tf.math.multiply(
                    tf.keras.backend.mean(self.loss_norm(y_true_yuv - y_pred_yuv), axis=[0, 1, 2]),
                    tf.constant([1., 2., 2], dtype=tf.float32)))

        return yuv_mse_loss


    def loss_fn_mse_rgb_yuv(self, y_true, y_pred):
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)

        rgb_mse_loss = tf.keras.backend.mean(self.loss_norm(y_true - y_pred))
        yuv_mse_loss = tf.keras.backend.mean(
                tf.math.multiply(
                    tf.keras.backend.mean(self.loss_norm(y_true_yuv - y_pred_yuv), axis=[0, 1, 2]),
                    tf.constant([1., 2., 2], dtype=tf.float32)))

        return rgb_mse_loss + yuv_mse_loss


    def loss_fn_ploss(self, y_true, y_pred):
        vgg16_perceptual_loss = 0

        y_true  = y_true / (self.input_max/2) - 1
        y_pred  = y_pred / (self.input_max/2) - 1
        for loss_model in self.loss_model:
            vgg16_perceptual_loss += tf.keras.backend.mean(
                    self.loss_norm(loss_model(y_true) - loss_model(y_pred)))

        return vgg16_perceptual_loss * self.ploss_scaling_factor


    def loss_fn_ssim(self, y_true, y_pred):
        ssim_loss = 1. - tf.image.ssim(y_true, y_pred, 1)
        return ssim_loss

    def loss_fn_bayer(self, y_true, y_pred):

        y_true = self.get_patternized_1ch_raw_image(y_true)
        y_pred = self.get_patternized_1ch_raw_image(y_pred)

        bayer_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))


        return bayer_loss

def get_checkpoint_manager(model, optimizer, directory, name):
    checkpoint = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                    directory=directory,
                                                    max_to_keep=5,
                                                    checkpoint_name=name,
                                                    )
    return checkpoint_manager


class BwCkptCallback(Callback):
    def __init__(self, model, optimizer, directory, name ):
        super().__init__()
        self.manager = get_checkpoint_manager(model, optimizer, directory, name)
        self.val_loss = np.inf


    def set_model(self, model):
        self.model = model

    def on_train_begin(self, _):
         pass

    def on_epoch_end(self, epoch, logs={}):
        print('--------------------> save gogo START')
        self.manager.save()
        print('--------------------> save gogo END')



class TensorBoardImage(Callback):
    def __init__(self, log_dir, dataloader, patch_size, cnt_viz, input_bias, cfa_pattern=1):
        super().__init__()
        self.log_dir = log_dir
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.cnt_viz = cnt_viz
        self.input_bias = input_bias



        self.idx_R = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G1 = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G2 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_B = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(tag='Original Image', epoch=0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, tag, epoch):
        gidx = 0
        for idx,  (x, y) in enumerate(self.dataloader):
            pred   = self.model(x)
            diff   = tf.math.abs(y-pred)

            if self.input_bias:
                x    = (   x + 1) / 2
                y    = (   y + 1) / 2
                pred = (pred + 1) / 2
                diff /= 2

            if x.shape[-1] == 1:
                tf_R  = tf.constant(self.idx_R,  dtype=np.float32)
                tf_G  = tf.constant(self.idx_G1, dtype=np.float32)
                tf_B  = tf.constant(self.idx_B,  dtype=np.float32)


                R = tf.math.multiply(x[...,0], tf_R)
                G = tf.math.multiply(x[...,0], tf_G)
                B = tf.math.multiply(x[...,0], tf_B)


                x = tf.concat((tf.expand_dims(R, axis=-1),
                               tf.expand_dims(G, axis=-1),
                               tf.expand_dims(B, axis=-1)),
                                axis=-1)


            print('x.shape', x.shape)
            print('y.shape', y.shape)
            print('y.shape', y.shape)
            print('y.shape', y.shape)

            all_images = tf.concat( [tf.concat([x, y]      , axis=2),
                                     tf.concat([diff, pred], axis=2)] , axis=1)

            print('x: %.2f, %.2f' %( tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy()), end='')
            print(', y: %.2f, %.2f' %( tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy()), end='')
            print(', pred: %.2f, %.2f' % (  tf.reduce_min(pred).numpy(), tf.reduce_max(pred).numpy()), end='')
            print(', diff: %.2f, %.2f' % ( tf.reduce_min(diff).numpy(), tf.reduce_max(diff).numpy()))

            with self.writer.as_default():
                tf.summary.image(f"Viz set {gidx}", all_images, max_outputs=16, step=epoch)
            gidx+=1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs={}):
        self.write_image('Images', epoch)

def get_training_callbacks(names, base_path, model_name=None, dataloader=None, patch_size=128, cnt_viz=4, input_bias=True, initial_value_threshold=np.inf):
    callbacks=[]
    # callbacks =  tf.keras.callbacks.CallbackList()
    if 'ckeckpoint' in names:
        ckpt_dir = os.path.join(base_path, 'checkpoint', model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckeckpoint_dir = os.path.join(ckpt_dir, '{epoch:05d}_%s_{loss:.5e}.h5' % (model_name))
        callback_ckpt = tf.keras.callbacks.ModelCheckpoint(
                                filepath = ckeckpoint_dir,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                initial_value_threshold=initial_value_threshold )
        callbacks.append(callback_ckpt)
    if 'tensorboard' in names:
        tb_dir = os.path.join(base_path, 'board', model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_tb = tf.keras.callbacks.TensorBoard( log_dir=tb_dir,
                                                       histogram_freq=10,
                                                       write_graph=True,
                                                       write_images=False)
        callbacks.append(callback_tb)
    if 'image' in names:
        tb_dir = os.path.join(base_path, 'board', model_name, 'image') #, model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_images =TensorBoardImage( log_dir=tb_dir,
                                           dataloader = dataloader,
                                           patch_size = patch_size,
                                           cnt_viz = cnt_viz,
                                           input_bias=input_bias)
        callbacks.append(callback_images)
    return callbacks



def get_scheduler(type='cosine', lr_init=2e-3, lr_last=1e-5, steps=100):
    print('-------------------> scheduler type, ', type)

    if float(tf.__version__[:-2]) > 2.4:
        if type.lower() == 'cosine':
            print('-------------------> GOOD scheduler type1, ', type)
            scheduler =tf.keras.optimizers.schedules.CosineDecay(
                                initial_learning_rate=lr_init,
                                decay_steps=steps,
                                alpha=0.5,
                                name='CosineDecay')
        else:
            print('-------------------> WTF scheduler type1, ', type, type =='cosine')
            scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                initial_learning_rate=lr_init,
                                first_decay_steps=steps,
                                t_mul=2.0,
                                m_mul=1.0,
                                alpha=lr_last,
                                name='CosineDecayRestarts')
    else:
        if type.lower() == 'cosine':
            print('-------------------> GOOD scheduler type2, ', type)
            scheduler =tf.keras.experimental.CosineDecay(
                                initial_learning_rate=lr_init,
                                decay_steps=steps,
                                alpha=0.5,
                                name='CosineDecay')
        else:
            print('-------------------> WTF scheduler type2, ', type)
            scheduler = tf.keras.experimental.CosineDecayRestarts(
                                initial_learning_rate=lr_init,
                                first_decay_steps=steps,
                                t_mul=2.0,
                                m_mul=1.0,
                                alpha=lr_last,
                                name='CosineDecayRestarts')


    lr_callback =  tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    return lr_callback




if __name__ == '__main__':

    butils = bwutils('shrink', 'tetra')



    print('done')
