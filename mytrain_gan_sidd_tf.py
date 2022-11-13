


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from mymodel_tf import save_as_tflite, GenerationTF
from myutils_tf import *

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# os.environ["CUDA_VISIBLE_DEVICES"]='-1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NGPU = len(get_available_gpus())
if NGPU == 0:
    NGPU = 1

MODEL_NAME = __file__.split('.')[0]  # 'model_tetra_out_model_tetra_12ch'



# for Adam
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-8
INTERVAL = 600

###########################################################################
# Weights initializer for the layers.
kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    # x = ReflectionPadding2D()(input_tensor)
    x = input_tensor
    x = tf.keras.layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    # x = ReflectionPadding2D()(x)
    x = tf.keras.layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = tf.keras.layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

###########################################################################
"""
## Build the generators

The generator consists of downsampling blocks: nine residual blocks
and upsampling blocks. The structure of the generator is the following:

```
c7s1-64 ==> Conv block with `relu` activation, filter size of 7
d128 ====|
         |-> 2 downsampling blocks
d256 ====|
R256 ====|
R256     |
R256     |
R256     |
R256     |-> 9 residual blocks
R256     |
R256     |
R256     |
R256 ====|
u128 ====|
         |-> 2 upsampling blocks
u64  ====|
c7s1-3 => Last conv block with `tanh` activation, filter size of 7.
```
"""
def get_resnet_generator(
    filters=64,
    input_img_size=(128,128,1),
    output_channels=3,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = tf.keras.layers.Input(shape=input_img_size, name=name + "_img_input")
    # x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = img_input
    x = tf.keras.layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, padding='same', use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=tf.keras.layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=tf.keras.layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=tf.keras.layers.Activation("relu"))

    # Final block
    # x = ReflectionPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(output_channels, (7, 7), padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    model = tf.keras.models.Model(img_input, x, name=name)
    return model

###########################################################################
"""
## Build the discriminators

The discriminators implement the following architecture:
`C64->C128->C256->C512`
"""
def get_discriminator(
    filters=64,
    input_img_size=(128,128,3),
    kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = tf.keras.layers.Input(shape=input_img_size, name=name + "_img_input")
    x = tf.keras.layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=tf.keras.layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=tf.keras.layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = tf.keras.layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = tf.keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model
###########################################################################
class Gan(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        discriminator_D,
        lambda_l1=100.0,
    ):
        super(Gan, self).__init__()
        self.gen_G = generator_G
        self.disc_D = discriminator_D

    def call(self, inputs):
        return (
            self.disc_D(inputs),
            self.gen_G(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        disc_D_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        bayer_loss_fn,
    ):
        super(Gan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.disc_D_optimizer = disc_D_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        # self.cycle_loss_fn    = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        # self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        # self.cycle_loss_fn    = tf.keras.losses.MeanAbsoluteError( )
        # self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError( )

        # self.cycle_loss_fn    = abs_loss_fn
        self.abs_loss_fn = abs_loss_fn
        # self.identity_loss_fn = bayer_loss_fn

        self.bayer_loss_fn = bayer_loss_fn

    def save_models(self, base_path:str, name:str, include_optimizer:bool=False):
        prefix = ['G', 'D']
        ns = name.split('_')
        net = [self.gen_G, self.disc_D]
        for p, n in zip(prefix, net):
            pre = ns[0] + '_' + p
            new_name = '_'.join([pre] + ns[1:])
            new_name = os.path.join(base_path, new_name)
            n.save(new_name, include_optimizer=False)


    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        print('--------------------------')
        print('--------------------------')
        print('real_x.shape ', real_x.shape)
        print('real_y.shape ', real_y.shape)
        print('--------------------------')
        print('--------------------------')

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True) # sRGB -> RAW


            # Discriminator output

            disc_real_y = self.disc_D(real_y, training=True)
            disc_fake_y = self.disc_D(fake_y, training=True)

            # Generator loss
            gen_G_total_loss, gen_G_gan_loss, gen_G_l1_loss = \
                    self.generator_loss_fn(disc_fake_y, fake_y, real_y)

            # Discriminator loss
            disc_D_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        gen_G_grads = tape.gradient(gen_G_total_loss, self.gen_G.trainable_variables)

        # Get the gradients for the discriminators
        disc_D_grads = tape.gradient(disc_D_loss, self.disc_D.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(gen_G_grads, self.gen_G.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_D_optimizer.apply_gradients(
            zip(disc_D_grads, self.disc_D.trainable_variables)
        )

        return {
            "G_loss": gen_G_total_loss,
            "G_gan_loss": gen_G_gan_loss,
            "G_1l_loss": gen_G_l1_loss,
            "D_loss": disc_D_loss,
        }

###########################################################################

class TensorBoardImage(tf.keras.callbacks.Callback):
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
            # x: sRGB
            # y: RAW (1ch to 3ch, nonshrink)
            pred = self.model.gen_G(x)
            diff   = tf.math.abs(y-pred)

            if self.input_bias:
                x    = (   x + 1) / 2
                y    = (   y + 1) / 2
                pred = (pred + 1) / 2
                diff /= 2


            # print('x.shape', x.shape)
            # print('y.shape', y.shape)
            # print('y_fake.shape', y_fake.shape)
            # print('x_fake.shape', x_fake.shape)

            print('x: %.2f, %.2f' %( tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy()), end='')
            print(', y: %.2f, %.2f' %( tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy()), end='')
            print(', pred: %.2f, %.2f' % (  tf.reduce_min(pred).numpy(), tf.reduce_max(pred).numpy()), end='')
            print(', diff: %.2f, %.2f' % ( tf.reduce_min(diff).numpy(), tf.reduce_max(diff).numpy()))

            all_images = tf.concat( [tf.concat([x, y]      , axis=2),
                                     tf.concat([diff, pred], axis=2)] , axis=1)


            with self.writer.as_default():
                tf.summary.image(f"Viz set {gidx}", all_images, max_outputs=16, step=epoch)
            gidx+=1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs={}):
        self.write_image('Images', epoch)




class SaveModelH5(tf.keras.callbacks.Callback):

    def __init__(self, path, name, initial_value_threshold):
        super().__init__()
        self.path = path
        self.name = name
        self.initial_value_threshold=initial_value_threshold

    def on_train_begin(self, logs=None):
         self.val_loss = []
         self.val_loss.append(self.initial_value_threshold)


    def on_epoch_end(self, epoch, logs=None):
        # G_loss: 15.9404 - F_loss: 8.1391 - D_X_loss: 0.4872 - D_Y_loss:
        G_loss = logs.get("G_loss")
        current_val_loss = G_loss
        self.val_loss.append(current_val_loss)
        if current_val_loss <= min(self.val_loss):
            print('Find lowest val_loss. Saving entire model.')
            path = os.path.join(self.path, '{epoch:05d}_%s_{loss:.5e}.h5' % (self.name))
            print(path)

            os.makedirs(self.path, exist_ok=True)
            self.model.save_models(self.path,  '%05d_%s_%.5e.h5' % (epoch, self.name, current_val_loss), include_optimizer=True)

            # spath = os.path.join(self.path, f'{epoch:05d}_%s_{current_val_loss:.5e}' % (self.name))
            # os.makedirs(spath, exist_ok=True)
            # self.model.save(spath, save_format="tf") # < ----- Here
            # exit()



###########################################################################
# adv_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
def abs_loss_fn(y_true, y_pred):
        myloss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))
        return myloss
def mse_loss_fn(y_true, y_pred):
        myloss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
        return myloss

# def adv_loss_fn(y_true, y_pred):
#         myloss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
#         return myloss

# adv_loss_fn = tf.keras.losses.MeanSquaredError()
# def generator_loss_fn(fake):
#     fake_loss = mse_loss_fn(tf.ones_like(fake), fake)
#     return fake_loss


def generator_loss_fn(disc_gen_output, fake, target):
    LAMBDA = 100

    gan_loss = mse_loss_fn(tf.ones_like(disc_gen_output), disc_gen_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - fake))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# Define the loss function for the discriminators
# def discriminator_loss_fn(real, fake):
#     real_loss = mse_loss_fn(tf.ones_like(real), real)
#     fake_loss = mse_loss_fn(tf.zeros_like(fake), fake)
#     return (real_loss + fake_loss) * 0.5
def discriminator_loss_fn(real, fake):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

###########################################################################


# def load_models(model, path):

def load_model_if_exists(model, model_dir, model_name=None, ckpt_name=None):
    prev_epoch = 0
    prev_loss = np.inf
    if os.path.exists(model_dir):

        prefix = ['G', 'F', 'X', 'Y']
        net = [model.gen_G, model.gen_F, model.disc_X, model.disc_Y]

        for p, n in zip(prefix, net):
            weights = glob.glob(os.path.join(model_dir, f'*_{p}_*.h5' ))
            if len(weights) > 0:
                weights.sort()
                best_weight = weights[-1]
                print('---------------------> ', best_weight)
                n.load_weights(best_weight)
                idx = best_weight[len(model_dir):]
                prev_epoch = int(best_weight[len(model_dir)+1:len(model_dir)+6])
                prev_loss = float(best_weight.split('_')[-1][:-3])
                print('prev epoch', prev_epoch, ', prev_loss:', prev_loss)
            else:
                print('===========> TRAINED WEIGHTS NOT EXIST', len(weights))
    return model, prev_epoch, prev_loss


##########################################################################




###########################################################################



def main(args):

    input_type = 'rgb'

    # update params. using input arguments
    input_type = args.input_type
    patch_size =  args.patch_size
    batch_size = args.batch_size
    constraint_max = args.constraint_max
    input_max = args.input_max
    constraint = {'min_value': 0, 'max_value': constraint_max}
    model_name = args.model_name
    data_path = args.data_path
    model_sig = args.model_sig
    myepoch = args.epoch

    print('test, ', args.test, type(args.test))
    # exit()

    # loss_type = ['rgb', 'yuv', 'ploss'] # 'rgb', 'yuv', 'ploss'
    loss_type = ['rgb', 'yuv', 'ssim']  # 'rgb', 'yuv', 'ploss
    # loss_type = ['rgb']  # 'rgb', 'yuv', 'ploss
    # loss_type = ['yuv']

    # get util class
    if args.test:
        cache_enable=False
    else:
        cache_enable=True

    utils = bwutils(input_type,
                    cfa_pattern='bayer',
                    patch_size=patch_size,
                    crop_size=patch_size,
                    input_max=input_max,
                    use_unprocess=False,
                    loss_type=loss_type, # 'rgb', 'yuv', 'ploss'
                    loss_mode='2norm',
                    loss_scale=1e4,
                    cache_enable=cache_enable)

    base_path = 'model_dir'
    if not os.path.exists(base_path):
        print(">>>>>>>>>>>>>> ", os.path.exists(base_path))
        exit()
        os.makedirs(base_path, exist_ok=True)

    model_dir = os.path.join(base_path, 'checkpoint', "gan_" + model_name + model_sig)

    ## dataset
    if args.test:
        data_path = 'datasets/sidd/tfrecords'
        data_path = "N:/dataset/SIDD/small/tfrecords"

    def get_tfrecords(path, keyword):
        files = tf.io.gfile.glob(os.path.join(path, f'*{keyword}*tfrecords'))
        files.sort()
        return files
    train_files = get_tfrecords(data_path, 'train')
    eval_files  = get_tfrecords(data_path, 'valid')
    viz_files   = get_tfrecords(data_path, 'viz')

    # if args.test:
    #     data_path = 'datasets/sidd/tfrecords'
    #     train_files = get_tfrecords(data_path, 'train*S6_noisy')
    #     eval_files = get_tfrecords(data_path, 'test*IP')
    #     viz_files = get_tfrecords(data_path, 'viz')


    print('data_path, ', data_path)
    print('\n'.join(train_files))
    print('\n'.join(eval_files))
    print('\n'.join(viz_files))

    ## training params setup
    print('=========================================================')
    print('=========================================================')
    print('=========================================================')
    print('=========================================================')
    print('=========================================================')
    print('========================================================= NGPU', NGPU)


    # exit()

    if args.test:
        batch_size = 1
    batch_size_train = batch_size * NGPU  # 128
    batch_size_eval  = batch_size * NGPU
    batch_size_viz   = batch_size  # 128
    # batch_size      = 32
    # batch_size_eval = 32
    batch_size_viz  = 8
    print('batch_size: ', batch_size_train, batch_size_eval, batch_size_viz)
    #exit()
    train_params = {'filenames': train_files,
                    'mode': tf.estimator.ModeKeys.TRAIN,
                    'threads': 2,
                    'shuffle_buff': 1024,
                    'batch': batch_size_train,
                    'input_type':input_type,
                    'train_type': 'unprocessing'
                    }
    eval_params = {'filenames': eval_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 1024,
                   'batch': batch_size_eval,
                   'input_type': input_type,
                   'train_type': 'unprocessing'
                   }

    viz_params = {'filenames': viz_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 1024,
                   'batch': batch_size_viz,
                   'input_type': input_type,
                   'train_type': 'unprocessing'
                   }

    dataset_train = utils.dataset_input_fn_cycle(train_params)
    dataset_eval  = utils.dataset_input_fn_cycle(eval_params)
    dataset_viz   = utils.dataset_input_fn_cycle(viz_params)

    # print('train set len : ', tf.data.experimental.cardinality(dataset_train))
    # print('train set len : ', dataset_train.element_spec)

    # print(len(list(dataset_train)), len(list(dataset_eval)), len(list(dataset_viz)))
    # exit()

    cnt_train, cnt_valid = 2540*10, 2580 # with noise
    if args.test:
        cnt_train, cnt_valid = 8, 8 # for test

    cnt_viz = 8



    # ##################################################
    # ##################################################
    # ds_viz = next(iter(dataset_viz))

    # srgb, raw = ds_viz
    # print(type(srgb.numpy()), type(raw.numpy()))
    # print((srgb.numpy().shape), (raw.numpy()).shape)
    # plt.figure(figsize=(16,16))
    # for idx, (s, r) in enumerate(zip(srgb, raw)):
    #     s = (s + 1)/2
    #     r = (r + 1) / 2

    #     plt.subplot(8,2,idx*2+1)
    #     plt.imshow(s)
    #     plt.subplot(8,2, idx*2+2)
    #     plt.imshow(r)
    #     print(idx, np.amin(s), np.amax(s), np.amin(r), np.amax(r))


    # plt.show()
    # exit()
    # ##################################################
    # ##################################################

    #########################
    ## training gogo

    if True:
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

        if input_type not in ['shrink', 'nonshrink', 'nonshrink_4ch', 'rgb']:
            raise ValueError('unkown input_type, ', input_type)

        #####################
        ## Get model gogo
        #####################

        ########################################
        # Get the generator
        bG = GenerationTF(model_name =  model_name,
                          input_shape=(patch_size, patch_size, 3),
                          kernel_regularizer=True,
                          kernel_constraint=constraint_max)
        gen_G = bG.model

        # Get the discriminator
        disc_D = get_discriminator(name="discriminator_D")  # disc for G : RAW

        # Create GAN model
        gan_model = Gan(generator_G=gen_G, discriminator_D=disc_D)


        # Compile the model
        gan_model.compile(
            gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            disc_D_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            gen_loss_fn=generator_loss_fn,
            disc_loss_fn=discriminator_loss_fn,
            bayer_loss_fn=utils.loss_fn_bayer
        )
        print('model created done done')
        ########################################

        # load pre-trained model
        prev_epoch = 0
        gan_model, prev_epoch, prev_loss = load_model_if_exists(gan_model, model_dir)


        model = gan_model
        if False:
            model.input.set_shape(1 + model.input.shape[1:]) # to freeze model
        model.save_models(os.path.join(base_path, 'checkpoint' ), f'gan_{model_name}_model_structure.h5', include_optimizer=False)
        # model.summaries()




        ## callbacks for training loop
        tb_dir = os.path.join(base_path, 'board', "gan_" + model_name + model_sig, 'image') #, model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_images =TensorBoardImage( log_dir=tb_dir,
                                                dataloader=dataset_viz,
                                                patch_size=patch_size,
                                                cnt_viz=5,
                                                input_bias=True)

        ckpt_dir = os.path.join(base_path, 'checkpoint', "gan_" + model_name + model_sig)
        callback_ckpt = SaveModelH5(path=ckpt_dir, name="gan_" + model_name + model_sig, initial_value_threshold=prev_loss)



        callbacks = get_training_callbacks(['tensorboard'],
                                            base_path=base_path, model_name="gan_" + model_name + model_sig,
                                            dataloader=dataset_viz, cnt_viz=cnt_viz)
        ## lr callback
        callback_lr = get_scheduler(type='cosine', lr_init=LEARNING_RATE, steps=myepoch)
        callbacks.append(callback_lr)
        callbacks.append(callback_images)
        callbacks.append(callback_ckpt)


        # exit()
        # train gogo
        more_ckpt_ratio = 2
        model.fit(dataset_train,
                    epochs=myepoch*more_ckpt_ratio,
                    steps_per_epoch=(cnt_train // (batch_size*more_ckpt_ratio)) + 1,
                    initial_epoch=prev_epoch+1,
                    validation_data=dataset_eval,
                    validation_steps=cnt_valid // batch_size_eval,
                    validation_freq=1,
                    callbacks=callbacks
                    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--input_type',
            type=str,
            default='rgb',
            help='shrink / nonshrink / nonshrink_4ch. default:nonshrink_4ch')

    parser.add_argument(
            '--input_max',
            type=float,
            default=1.,
            help='input_max')

    parser.add_argument(
            '--constraint_max',
            type=float,
            default=6,
            help='maximum constraint value for kernel/bias')

    parser.add_argument(
            '--batch_size',
            type=int,
            default=16,
            help='input patch size')

    parser.add_argument(
            '--epoch',
            type=int,
            default=600,
            help='epoch')

    parser.add_argument(
            '--patch_size',
            type=int,
            default=128,
            help='input patch size')

    parser.add_argument(
            '--model_name',
            type=str,
            default='unet',
            help='resnet_flat, resnet_ed, bwunet, unet, unetv2')

    parser.add_argument(
            '--model_sig',
            type=str,
            default='_newstart',
            help='model postfix')

    parser.add_argument(
            '--data_path',
            type=str,
            default='/dataset/SIDD/small/tfrecords',
            help='add noise on dec input')

    parser.add_argument(
            '--test',
            type=bool,
            default=False,
            help='test')

    args = parser.parse_args()

    main(args)
