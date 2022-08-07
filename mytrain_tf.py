


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import glob
import datetime
import numpy as np


from myutils_tf import bwutils
# from tensorflow.keras import backend as K
import  tensorflow.keras.backend as K

from mymodel_tf import save_as_tflite, GenerationTF


# To use limit number of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# tf.compat.v1.disable_eager_execution()
TF_VER=1 if tf.__version__.split('.')[0]=='1' else (2 if tf.__version__.split('.')[0]=='2' else None)

# if TF_VER == 2:
#     tf.compat.v1.disable_eager_execution()

def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1,2,3))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NGPU = len(get_available_gpus())
if NGPU == 0:
    NGPU = 1

MODEL_NAME = __file__.split('.')[0]  # 'model_tetra_out_model_tetra_12ch'



# for Adam
LEARNING_RATE = 1e-4
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
    use_bn = args.use_bn
    add_noise_dec_input = args.add_noise_dec_input



    # loss_type = ['rgb', 'yuv', 'ploss'] # 'rgb', 'yuv', 'ploss'
    loss_type = ['rgb', 'yuv']  # 'rgb', 'yuv', 'ploss
    loss_type = ['rgb']  # 'rgb', 'yuv', 'ploss

    # get util class
    utils = bwutils(input_type,
                    cfa_pattern='tetra',
                    patch_size=patch_size,
                    crop_size=patch_size,
                    input_max=input_max,
                    loss_type=loss_type, # 'rgb', 'yuv', 'ploss'
                    loss_mode='2norm',
                    cache_enable=False)


    ## model file and model_dir
    model_file_name = f'{model_name}_{patch_size}x{patch_size}'


    model_dir = 'model_dir_' + model_file_name
    print('model_dir = ', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir + '/models', exist_ok=True)

    model_file_full_path = os.path.join(model_dir, model_file_name)

    ## dataset
    data_path = '/dataset/MIT/tfrecords_data_only'
    data_path = 'datasets/pixelshift/tfrecords'
    # data_path = '/home/team19/datasets/pixelshift/tfrecords'

    def get_tfrecords(path, keyword):
        files = tf.io.gfile.glob(os.path.join(path, f'*{keyword}*tfrecords'))
        files.sort()
        return files
    train_files = get_tfrecords(data_path, 'train')
    eval_files = get_tfrecords(data_path, 'valid')
    viz_files = get_tfrecords(data_path, 'viz10')


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
    batch_size = 1 * NGPU # 128
    batch_size = 1  # 128
    batch_size_eval = 1 * 1 * NGPU
    batch_size_viz = 4  # 128
    train_params = {'filenames': train_files,
                    'mode': tf.estimator.ModeKeys.TRAIN,
                    'threads': 4,
                    'shuffle_buff': 100,
                    'batch': batch_size,
                    'input_type':input_type
                    }
    eval_params = {'filenames': eval_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 4,
                   'shuffle_buff': 100,
                   'batch': batch_size_eval,
                   'input_type': input_type}

    viz_params = {'filenames': viz_files,
                   'mode': tf.estimator.ModeKeys.EVAL,
                   'threads': 4,
                   'shuffle_buff': 100,
                   'batch': batch_size_viz,
                   'input_type': input_type}

    dataset_train = utils.dataset_input_fn(train_params)
    dataset_eval = utils.dataset_input_fn(eval_params)
    dataset_viz = utils.dataset_input_fn(viz_params)

    print('train set len : ', tf.data.experimental.cardinality(dataset_train))
    print('train set len : ', dataset_train.element_spec)


    cnt_train = 92800
    cnt_valid = 4800
    # cnt_train = 4
    # cnt_valid = 4
    cnt_viz = 4


    #########################
    ## training gogo
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():



        if input_type not in ['shrink', 'nonshrink', 'nonshrink_4ch', 'rgb']:
            raise ValueError('unkown input_type, ', input_type)



        #####################
        #####################
        #####################

        bw = GenerationTF(model_name =  model_name)





        model = bw.model

        model.input.set_shape(1 + model.input.shape[1:])




        save_as_tflite(model, f'model_{model_name}')
        model.summary()



        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam')
        model.compile(optimizer=optimizer,  # 'adam',
                    loss=utils.loss_fn,  # 'mse',
                    metrics=[utils.loss_fn])



        ## load designaged trained model
        trained_model_file_name = '01646_2.23672e-03.h5'


        trained_model_file_name = os.path.join(model_dir, trained_model_file_name)
        if os.path.isfile(trained_model_file_name):
            print('=============> load pre trained mode: ', trained_model_file_name)
            model.load_weights(trained_model_file_name)
        else:
            print('=============> pre trained model not exists ')
            # raise ValueError('unkown trained weight', trained_model_file_name)


        # load previous trained model if exist
        prev_epoch=0

        trained_weights = glob.glob(os.path.join(model_dir, 'models/*.h5' ))
        print('--=-=-=-> ', trained_weights)
        if len(trained_weights ) > 0:
            print('===========> %d TRAINED WEIGHTS EXIST' % len(trained_weights))
            trained_weights.sort()
            trained_weights = trained_weights[-1]
            model.load_weights(trained_weights)
            idx = trained_weights.rfind('models')
            prev_epoch = int(trained_weights[idx+7:idx+7+5])
            print('prev epoch', prev_epoch)
        else:
            print('===========> TRAINED WEIGHTS NOT EXIST', len(trained_weights))


        model.save(model_dir + '/model_structure.h5', include_optimizer=False)

        # call backs
        tensorboard_dir = os.path.join(model_dir, 'board')
        os.makedirs(tensorboard_dir, exist_ok=True)

        from myutils_tf import TensorBoardImage
        image_callback =TensorBoardImage( log_dir=os.path.join(model_dir, 'board'),
                                        dataloader = dataset_viz,
                                        patch_size = 128,
                                        cnt_viz = cnt_viz)

        callbacks = [  # SaveBestCallback(), # MyCustomCallback(),
            image_callback,
            tf.keras.callbacks.TensorBoard(log_dir=model_dir + '/board',
                                        histogram_freq=2,
                                        write_graph=True,
                                        write_images=False,
                                        ),
            tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_dir + '/models/{epoch:05d}_%s_{loss:.5e}_1.h5' % (MODEL_NAME),
                    # the `val_loss` score has improved.
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_loss',  # 'val_mse_yuv_loss',
                    verbose=1)
        ]

        more_ckpt_ratio = 1
        model.fit(dataset_train,
                    epochs=INTERVAL*more_ckpt_ratio,
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
            default=16384,
            help='input_max')

    parser.add_argument(
            '--constraint_max',
            type=float,
            default=6,
            help='maximum constraint value for kernel/bias')

    parser.add_argument(
            '--batch_size',
            type=int,
            default=8,
            help='input patch size')

    parser.add_argument(
            '--patch_size',
            type=int,
            default=128,
            help='input patch size')

    parser.add_argument(
            '--use_bn',
            type=str,
            default='True',
            help='use batch normalization on the enc output')

    parser.add_argument(
            '--model_name',
            type=str,
            default='bwunet',
            help='Networ model name, ')

    parser.add_argument(
            '--add_noise_dec_input',
            type=str,
            default='True',
            help='add noise on dec input')

    args = parser.parse_args()

    main(args)


