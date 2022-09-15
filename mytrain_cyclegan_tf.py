


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
# os.environ["CUDA_VISIBLE_DEVICES"]='6'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



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
class CycleGan(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def call(self, inputs):
        return (
            self.disc_X(inputs),
            self.disc_Y(inputs),
            self.gen_G(inputs),
            self.gen_F(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        bayer_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        # self.cycle_loss_fn    = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        # self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        # self.cycle_loss_fn    = tf.keras.losses.MeanAbsoluteError( )
        # self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError( )

        self.cycle_loss_fn    = bayer_loss_fn
        self.identity_loss_fn = bayer_loss_fn

        self.bayer_loss_fn = bayer_loss_fn

    def save_models(self, base_path:str, name:str, include_optimizer:bool=False):
        prefix = ['G', 'F', 'X', 'Y']
        ns = name.split('_')
        net = [self.gen_G, self.gen_F, self.disc_X, self.disc_Y]
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
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True) # RAW -> sRGB

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

###########################################################################

class TensorBoardImageCycle(Callback):
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
            y_fake = self.model.gen_G(x)
            x_fake = self.model.gen_F(y)

            if self.input_bias:
                x    = (   x + 1) / 2
                y    = (   y + 1) / 2
                y_fake = (y_fake + 1) / 2
                x_fake = (x_fake + 1) / 2




            # print('x.shape', x.shape)
            # print('y.shape', y.shape)
            # print('y_fake.shape', y_fake.shape)
            # print('x_fake.shape', x_fake.shape)

            print('x: %.2f, %.2f' %( tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy()), end='')
            print(', y: %.2f, %.2f' %( tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy()), end='')
            print(', x_fake: %.2f, %.2f' % (  tf.reduce_min(x_fake).numpy(), tf.reduce_max(x_fake).numpy()), end='')
            print(', y_fake: %.2f, %.2f' % ( tf.reduce_min(y_fake).numpy(), tf.reduce_max(y_fake).numpy()))

            all_images = tf.concat( [tf.concat([x, y]      , axis=2),
                                     tf.concat([x_fake, y_fake], axis=2)] , axis=1)


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
        F_loss = logs.get("F_loss")
        current_val_loss = G_loss + F_loss
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
def adv_loss_fn(y_true, y_pred):
        myloss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))
        return myloss


# adv_loss_fn = tf.keras.losses.MeanSquaredError()
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5
###########################################################################


# def load_models(model, path):

def load_model_if_exists(model, model_dir, model_name, ckpt_name=None):
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
                    cfa_pattern='tetra',
                    patch_size=patch_size,
                    crop_size=patch_size,
                    input_max=input_max,
                    use_unprocess=True,
                    loss_type=loss_type, # 'rgb', 'yuv', 'ploss'
                    loss_mode='2norm',
                    loss_scale=1e4,
                    cache_enable=cache_enable)







    base_path = 'model_dir'
    os.makedirs(base_path, exist_ok=True)
    model_dir = os.path.join(base_path, 'checkpoint', model_name + model_sig)



    ## dataset
    if args.test:
        data_path = 'datasets/sidd/tfrecords'
    def get_tfrecords(path, keyword):
        files = tf.io.gfile.glob(os.path.join(path, f'*{keyword}*tfrecords'))
        files.sort()
        return files
    train_files = get_tfrecords(data_path, 'train*S6_noisy')
    eval_files = get_tfrecords(data_path, 'test*IP')
    viz_files = get_tfrecords(data_path, 'viz')

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
    batch_size_viz  = 5
    print('batch_size: ', batch_size_train, batch_size_eval, batch_size_viz)
    #exit()
    train_params = {'filenames': train_files,
                    'mode': tf.estimator.ModeKeys.TRAIN,
                    'threads': 2,
                    'shuffle_buff': 256,
                    'batch': batch_size_train,
                    'input_type':input_type,
                    'train_type': 'unprocessing'
                    }
    eval_params = {'filenames': eval_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 256,
                   'batch': batch_size_eval,
                   'input_type': input_type,
                   'train_type': 'unprocessing'
                   }

    viz_params = {'filenames': viz_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 256,
                   'batch': batch_size_viz,
                   'input_type': input_type,
                   'train_type': 'unprocessing'
                   }

    dataset_train = utils.dataset_input_fn_cycle(train_params)
    dataset_eval = utils.dataset_input_fn_cycle(eval_params)
    dataset_viz = utils.dataset_input_fn_cycle(viz_params)

    # print('train set len : ', tf.data.experimental.cardinality(dataset_train))
    # print('train set len : ', dataset_train.element_spec)

    # print(len(list(dataset_train)), len(list(dataset_eval)), len(list(dataset_viz)))
    # exit()

    cnt_train, cnt_valid = 92800, 4800 # w/o noise
    cnt_train, cnt_valid = 127020*2//2//6, 32640//6 # with noise
    if args.test:
        cnt_train, cnt_valid = 8, 8 # for test

    cnt_viz = 10


    #########################
    ## training gogo

    # if True:

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        if input_type not in ['shrink', 'nonshrink', 'nonshrink_4ch', 'rgb']:
            raise ValueError('unkown input_type, ', input_type)

        #####################
        ## Get model gogo
        #####################

        ########################################
        # Get the generators
        gen_G = get_resnet_generator(name="generator_G",
                                    input_img_size=(patch_size,patch_size,3),
                                    output_channels=3) # sRGB to RAW

        gen_F = get_resnet_generator(name="generator_F",
                                    input_img_size=(patch_size,patch_size,3),
                                    output_channels=3) # RAW to sRGB

        # Get the discriminators
        disc_Y = get_discriminator(name="discriminator_Y")  # disc for G : RAW
        disc_X = get_discriminator(name="discriminator_X")  # disc for F : sRGB

        # Create cycle gan model
        cycle_gan_model = CycleGan(
            generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
        )


        # Compile the model
        cycle_gan_model.compile(
            gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            gen_loss_fn=generator_loss_fn,
            disc_loss_fn=discriminator_loss_fn,
            bayer_loss_fn=utils.loss_fn_bayer
        )
        print('model created done done')
        ########################################

        # load pre-trained model
        # model, prev_epoch, prev_loss = load_checkpoint_if_exists(model, model_dir, model_name, trained_model_file_name)

        prev_epoch = 0
        cycle_gan_model, prev_epoch, prev_loss = load_model_if_exists(cycle_gan_model,
                                                                      model_dir,
                                                                      model_name,
                                                                      prev_epoch)

        # bw = GenerationTF(model_name =  model_name, kernel_regularizer=True, kernel_constraint=True)

        model = cycle_gan_model
        if False:
            model.input.set_shape(1 + model.input.shape[1:]) # to freeze model
        # model.save(os.path.join(base_path, 'checkpoint' , f'{model_name}_model_structure.h5'), include_optimizer=False)
        model.save_models(os.path.join(base_path, 'checkpoint' ), f'{model_name}_model_structure.h5', include_optimizer=False)
        # model.summaries()




        ## callbacks for training loop
        tb_dir = os.path.join(base_path, 'board', model_name + model_sig, 'image') #, model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_images =TensorBoardImageCycle( log_dir=tb_dir,
                                                dataloader = dataset_viz,
                                                patch_size = patch_size,
                                                cnt_viz = 5,
                                                input_bias=True)

        ckpt_dir = os.path.join(base_path, 'checkpoint', model_name + model_sig)
        callback_ckpt = SaveModelH5(path=ckpt_dir, name=model_name + model_sig, initial_value_threshold=prev_loss)

        callbacks = get_training_callbacks(['tensorboard'],
                                            base_path=base_path, model_name=model_name + model_sig,
                                            dataloader=dataset_viz, cnt_viz=cnt_viz)
        ## lr callback
        callback_lr = get_scheduler(type='cosine', lr_init=LEARNING_RATE, steps=myepoch)
        callbacks.append(callback_lr)
        callbacks.append(callback_images)
        callbacks.append(callback_ckpt)

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
            default=65535,
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
            default='cyclegan',
            help='resnet_flat, resnet_ed, bwunet, unet, unetv2')

    parser.add_argument(
            '--model_sig',
            type=str,
            default='_bayer',
            help='model postfix')

    parser.add_argument(
            '--data_path',
            type=str,
            # default='/home/team19/datasets/sidd/tfrecords',
            default='/home/team01/datasets/sidd/tfrecords',
            help='add noise on dec input')

    parser.add_argument(
            '--test',
            type=bool,
            default=False,
            help='test')

    args = parser.parse_args()

    main(args)


