import os, glob, math
import numpy as np
import tensorflow as tf
from PIL import Image

from myutils_tf import bwutils


def save_4ch_to_3ch(path_pixel_shift):
    print(path_pixel_shift)

    files = glob.glob(os.path.join(path_pixel_shift, '*.npy'))
    # print(files)
    for idx, file in enumerate(files):
        if '3ch' not in file:
            arr = np.load(file)
            arr_3ch = arr[:,:,(0,1,3)]
            file_new = file[:-4] + '_3ch.npy'
            np.save(file_new, arr_3ch)


def get_model(model_name, model_sig):
    base_path = os.path.join('model_dir', 'checkpoint')
    structure_path = os.path.join(base_path, model_name + '_model_structure.h5')
    ckpt_path = os.path.join(base_path, model_name + '_' + model_sig)
    print(structure_path, '\n', ckpt_path)


    # load model structure
    model = tf.keras.models.load_model(structure_path)

    # find latest weights and load
    ckpts = glob.glob(os.path.join(ckpt_path, '*.h5'))
    ckpts.sort()
    ckpt = ckpts[-1]
    model.load_weights(ckpt)

    print(ckpt)
    # model.summary()
    return model

def normalize1_and_gamma(arr, bits=16, beta=1/2.2):
    arr = arr / (2**bits -1) # (0, 1)
    arr = arr ** beta   # (0, 1)
    return arr


def main():
    # model name
    model_name = 'demosaic'

    # model sig
    model_sig = 'mit'
    # get model
    model = get_model(model_name, model_sig)
    # model.summary()


    # cellsize
    if model_name in ['demosaic']:
        cell_size=1
        cfa_pattern = 'bayer'
    elif model_sig in ['tetra']:
        cell_size=2
        cfa_pattern = 'tetra'
    elif model_sig in ['sedec']:
        cell_size=4
        cfa_pattern = 'sedec'
    else:
        ValueError('Unknown model_name or model_sig', model_name, model_sig)
        exit()




    # test data
    PATH_PIXELSHIFT = 'C:/Users/AI38/datasets/pixelshfit/PixelShift200_test'
    files = glob.glob(os.path.join(PATH_PIXELSHIFT, '*_3ch.npy'))
    pad_size = 32
    patch_size = 128


    # utils for patternized
    utils = bwutils(input_type='rgb',
                        cfa_pattern=cfa_pattern,
                        patch_size=patch_size,
                        crop_size=patch_size,
                        input_max=65536,
                        use_unprocess=False,
                        loss_type=['rgb'],
                        loss_mode='2norm',
                        loss_scale=1e4,
                        cache_enable=False)




    # shape = np.load(files[0]).shape
    # height, width, channels = np.load(files[0]).shape
    # npatches_y, npatches_x = math.ceil(shape[0]/patch_size), math.ceil(shape[1]/patch_size)
    # print(arr_pred.shape)
    for idx, file in enumerate(files):
        arr = np.load(file)    # (0, 65535)
        arr = arr / (2**16 -1) # (0, 1)
        # arr = arr ** (1/2.2)   # (0, 1)



        print('arr.shape', arr.shape)
        arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size),(0, 0)), 'symmetric')
        print('arr.shape', arr.shape)

        height, width, channels = arr.shape
        npatches_y = math.ceil((height+2*pad_size) / (patch_size-2*pad_size))
        npatches_x = math.ceil((width +2*pad_size) / (patch_size-2*pad_size))


        arr_pred = np.zeros_like(arr)
        print(idx, file, arr.shape, arr_pred.shape)
        # exit()
        cnt=0
        tcnt= npatches_x*npatches_y
        for idx_y in range(npatches_y):
            for idx_x  in range(npatches_x):
                if(cnt%10==0):
                    print(f'{cnt} / {tcnt}')
                cnt+=1
                sy = idx_y * (patch_size-2*pad_size)
                ey = sy + patch_size
                sx = idx_x * (patch_size-2*pad_size)
                ex = sx + patch_size

                if ey >= height:
                    ey = height-1
                    sy = height-patch_size-1

                if ex >= width:
                    ex = width-1
                    sx = width-patch_size-1

                arr_patch = arr[sy:ey, sx:ex,:]
                arr_patch = utils.get_patternized_1ch_raw_image(arr_patch)

                print(np.amin(arr_patch), np.amax(arr_patch) )
                # exit()
                # # pre-process # no gamma & bais for demosaic/remosaic
                # arr_patch = arr_patch**(1/2.2)
                # arr_patch = (arr_patch*2) -1  # (0, 1) -> (-1, 1)

                # prediction
                pred = model.predict(arr_patch[np.newaxis,...])
                # print(pred.shape)

                # post-process
                arr_pred[sy+pad_size:ey-pad_size, sx+pad_size:ex-pad_size, :] = \
                            pred[0, pad_size:-pad_size, pad_size:-pad_size, :]
                            #  (pred[0, pad_size:-pad_size, pad_size:-pad_size, :]+1)/2 #  (-1, 1) -> (0, 1)
                # print(np.amin(arr_patch), np.amax(arr_patch), np.amin(arr_pred), np.amax(arr_pred))
                # exit()

        # arr_pred.astype(np.uint8)
        arr_pred = arr_pred[pad_size:-pad_size, pad_size:-pad_size, :]
        img_pred = Image.fromarray((arr_pred*255).astype(np.uint8))
        # name = os.path.join(PATH_PIXELSHIFT, f'inf_{model_name}_{model_sig}_%02d.png'%(idx+1))
        name = os.path.join(PATH_PIXELSHIFT, f'inf_{model_name}_{model_sig}_%02d.png'%(idx+1))
        img_pred.save(name)
        print(np.amin(img_pred), np.amax(img_pred), np.amin(arr_pred.astype(np.uint8)), np.amax(arr_pred.astype(np.uint8)))


        # exit()


if __name__ == '__main__':
    main()