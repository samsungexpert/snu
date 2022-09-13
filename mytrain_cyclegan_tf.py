


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

# os.environ["CUDA_VISIBLE_DEVICES"]='-1'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# To use limit number of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"



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
    input_img_size=(128,128,1),
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
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        # self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        # self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        self.cycle_loss_fn    = tf.keras.losses.MeanAbsoluteError( )
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError( )


    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

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
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

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
    train_files = get_tfrecords(data_path, 'viz')
    eval_files = get_tfrecords(data_path, 'viz')
    viz_files = get_tfrecords(data_path, 'viz')

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
    batch_size      = batch_size * NGPU  # 128
    batch_size_eval = batch_size * NGPU
    batch_size_viz  = batch_size  # 128
    batch_size_viz  = batch_size
    print('batch_size: ', batch_size, batch_size_eval, batch_size_viz)
    # exit()
    train_params = {'filenames': train_files,
                    'mode': tf.estimator.ModeKeys.TRAIN,
                    'threads': 2,
                    'shuffle_buff': 256,
                    'batch': batch_size,
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

    print(len(list(dataset_train)), len(list(dataset_eval)), len(list(dataset_viz)))
    exit()

    cnt_train, cnt_valid = 92800, 4800 # w/o noise
    cnt_train, cnt_valid = 96200, 4800 # with noise
    if args.test:
        cnt_train, cnt_valid = 8, 8 # for test

    cnt_viz = 10


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

        bw = GenerationTF(model_name =  model_name, kernel_regularizer=True, kernel_constraint=True)

        model = bw.model
        if False:
            model.input.set_shape(1 + model.input.shape[1:]) # to freeze model
        model.save(os.path.join(base_path, 'checkpoint' , f'{model_name}_model_structure.h5'), include_optimizer=False)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam')
        model.compile(optimizer=optimizer,  # 'adam',
                    loss=utils.loss_fn,  # 'mse',
                    metrics=[utils.loss_fn])


        ## load pre-trained model
        trained_model_file_name = '00003_resnet_flat_2.89940e-09.h5'
        model, prev_epoch, prev_loss = load_checkpoint_if_exists(model, model_dir, model_name, trained_model_file_name)



        ## callbacks for training loop
        callbacks = get_training_callbacks(['ckeckpoint', 'tensorboard', 'image'],
                                            base_path=base_path, model_name=model_name + model_sig,
                                            dataloader=dataset_viz, cnt_viz=cnt_viz, initial_value_threshold=prev_loss)
        ## lr callback
        callback_lr = get_scheduler(type='cosine', lr_init=LEARNING_RATE, steps=myepoch)
        callbacks.append(callback_lr)

        # train gogo
        more_ckpt_ratio = 1
        model.fit(dataset_train,
                    epochs=myepoch*more_ckpt_ratio,
                    steps_per_epoch=(cnt_train // (batch_size*more_ckpt_ratio)) + 1,
                    initial_epoch=prev_epoch,
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
            default=32,
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
            default='unetv2',
            help='resnet_flat, resnet_ed, bwunet, unet, unetv2')

    parser.add_argument(
            '--model_sig',
            type=str,
            default='_noise',
            help='model postfix')

    parser.add_argument(
            '--data_path',
            type=str,
            default='/home/team19/datasets/sidd/tfrecords',
        #     default='/data03/team01/pixelshift/tfrecords',
            help='add noise on dec input')

    parser.add_argument(
            '--test',
            type=bool,
            default=False,
            help='test')

    args = parser.parse_args()

    main(args)


