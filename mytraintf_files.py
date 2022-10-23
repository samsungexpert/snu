


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

from mymodel_tf import save_as_tflite, GenerationTF
from myutils_tf import *

# os.environ["CUDA_VISIBLE_DEVICES"]='-1'
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



def init_variables():
    sess = tf.compat.v1.keras.backend.get_session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)


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
    file_type = args.file_type

    loss_type = ['rgb', 'yuv', 'ssim']  # 'rgb', 'yuv', 'ploss

    # get util class
    if args.test:
        cache_enable=False
    else:
        cache_enable=True

    utils = bwutils(input_type,
                    cfa_pattern='tetra',
                    file_type=file_type,
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
        data_path = '/Users/bw/Datasets/mit'
    def get_image_files(path, keyword):
        p = os.path.join(path, keyword, '**', '*.png')
        files = glob.glob(p, recursive=True)
        # files.sort()
        return files
    train_files = get_image_files(data_path, 'val')
    eval_files = get_image_files(data_path, 'test')
    viz_files = get_image_files(data_path, 'train')

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

    if args.test:
        batch_size = 1
    batch_size      = batch_size * NGPU  # 128
    batch_size_eval = batch_size * NGPU
    batch_size_viz  = batch_size  # 128
    batch_size_viz  = 10
    print(batch_size, batch_size_eval, batch_size_viz)
    # exit()
    train_params = {'filenames': train_files,
                    'mode': tf.estimator.ModeKeys.TRAIN,
                    'threads': 2,
                    'shuffle_buff': 128,
                    'batch': batch_size,
                    'input_type':input_type,
                    'train_type': 'demosaic',
                    'file_type': file_type
                    }
    eval_params = {'filenames': eval_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 128,
                   'batch': batch_size_eval,
                   'input_type': input_type,
                   'train_type': 'demosaic',
                   'file_type': file_type
                   }

    viz_params = {'filenames': viz_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 2,
                   'shuffle_buff': 128,
                   'batch': batch_size_viz,
                   'input_type': input_type,
                   'train_type': 'demosaic',
                   'file_type': file_type
                   }

    dataset_train = utils.dataset_input_fn(train_params)
    dataset_eval = utils.dataset_input_fn(eval_params)
    dataset_viz = utils.dataset_input_fn(viz_params)


    cnt_train, cnt_valid = 53322, 6000
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
            default='nonshrink',
            help='shrink / nonshrink / nonshrink_4ch. default:nonshrink_4ch')

    parser.add_argument(
            '--input_max',
            type=float,
            default=255,
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
            default=200,
            help='epoch')

    parser.add_argument(
            '--patch_size',
            type=int,
            default=128,
            help='input patch size')

    parser.add_argument(
            '--model_name',
            type=str,
            default='remosaic',
            help='remosaic, resnet_flat, resnet_ed, bwunet, unet, unetv2')

    parser.add_argument(
            '--model_sig',
            type=str,
            default='_noise',
            help='model postfix')

    parser.add_argument(
            '--data_path',
            type=str,
            default='/content/drive/MyDrive/Datasets/mit/',
            help='add noise on dec input')

    parser.add_argument(
            '--file_type',
            type=str,
            default='png',
            help='file_type, default=png')

    parser.add_argument(
            '--test',
            type=bool,
            default=False,
            help='test')


    args = parser.parse_args()

    main(args)


