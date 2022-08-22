

import os
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

    trained_model_file_name = os.path.join(model_dir, ckpt_name)
    if (ckpt_name != None ) and \
       (os.path.isfile(trained_model_file_name) and os.path.exists(trained_model_file_name)):

        print('=============> load designated trained model: ', ckpt_name)
        model.load_weights(trained_model_file_name)
        idx = trained_weights.rfind(model_name)
        prev_epoch = int(trained_weights[idx-6:idx-1])

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
            print('prev epoch', prev_epoch)
        else:
            print('===========> TRAINED WEIGHTS NOT EXIST', len(trained_weights))

    return model, prev_epoch




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
        if cfa_pattern in ['tetra', 2]:
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


        if input_type in ['shrink_upscale']:
            print('before', self.idx_R.shape, self.idx_G1.shape, self.idx_G2.shape, self.idx_B.shape)
            self.idx_R = self.idx_R[:80, :80]
            self.idx_G1 = self.idx_G1[:80, :80]
            self.idx_G2 = self.idx_G2[:80, :80]
            self.idx_B = self.idx_B[:80, :80]
            self.idx_G = self.idx_G[:80, :80]
            print('after', self.idx_R.shape, self.idx_G1.shape, self.idx_G2.shape, self.idx_B.shape)


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

        elif input_type in ['shrink_upscale']:

            # todo: doenscale operation should be included
            crop_size_resized = int(self.crop_size // self.upscaling_factor)
            image_resized = self.resize_to_scale_factor(image, crop_size_resized)
            patternized = self.get_patternized_4ch_image(image_resized, is_shrink=True)

        elif input_type in ['nonshrink_4ch']:

            patternized = self.get_patternized_4ch_image(image, is_shrink=False)

        elif input_type in ['raw_1ch']:

            patternized = self.get_patternized_1ch_raw_image(image)

        elif input_type in ['nonshrink']:

            patternized = self.get_patternized_3ch_image(image)

        else:
            ValueError('unknown input type', input_type)

        return patternized, image


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

        std = 2 * (self.input_bits-10) * 24
        noise_gaussian = tf.random.normal(image.shape) * std
        noise_gaussian = self.scale_by_input_max(noise_gaussian)

        pstd = 8
        noise_poisson = tf.random.normal(image.shape) * tf.math.sqrt(image) * pstd
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




    def dataset_input_fn(self, params):
        dataset = tf.data.TFRecordDataset(params['filenames'])

        # Dataset cache need 120 Main memory
        if self.cache_enable is True:
            dataset = dataset.cache()

        dataset = dataset.map(partial(self.parse_tfrecord, mode=params['mode']), num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
        y_pred = y_pred / (self.input_max/2) - 1
        for loss_model in self.loss_model:
            vgg16_perceptual_loss += tf.keras.backend.mean(
                    self.loss_norm(loss_model(y_true) - loss_model(y_pred)))

        return vgg16_perceptual_loss * self.ploss_scaling_factor


    def loss_fn_ssim(self, y_true, y_pred):
        ssim_loss = 1. - tf.image.ssim(y_true, y_pred, 1)
        return ssim_loss




def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    _, height, width, channel = tensor.shape
    tensor = tensor[0]
    tensor_normalized = tensor - tensor.min()
    tensor_normalized /= tensor_normalized.max()
    tensor_normalized = img_as_ubyte(tensor_normalized)
    tensor_squeezed = np.squeeze(tensor_normalized)
    image = Image.fromarray(tensor_squeezed)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    summary = tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )
    return summary

class TensorBoardImage(Callback):
    def __init__(self, log_dir, dataloader, patch_size, cnt_viz, input_bias):
        super().__init__()
        self.log_dir = log_dir
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.cnt_viz = cnt_viz
        self.input_bias = input_bias

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

def get_training_callbacks(names, base_path, model_name=None, dataloader=None, patch_size=128, cnt_viz=4, input_bias=True ):
    callbacks=[]
    if 'ckeckpoint' in names:
        ckpt_dir = os.path.join(base_path, 'checkpoint', model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckeckpoint_dir = os.path.join(ckpt_dir, '{epoch:05d}_%s_{loss:.5e}_1.h5' % (model_name))
        callback_ckpt = tf.keras.callbacks.ModelCheckpoint(
                                filepath = ckeckpoint_dir,
                                # filepath=model_dir + '/models/{epoch:05d}_%s_{loss:.5e}_1.h5' % (MODEL_NAME),
                                # the `val_loss` score has improved.
                                save_best_only=True,
                                save_weights_only=False,
                                monitor='val_loss',  # 'val_mse_yuv_loss',
                                verbose=1)
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




if __name__ == '__main__':

    butils = bwutils('shrink', 'tetra')



    print('done')