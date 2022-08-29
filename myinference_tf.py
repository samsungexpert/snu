import os, glob, math
import numpy as np
import tensorflow as tf
from PIL import Image




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

    # print(ckpt)
    # model.summary()
    return model

def main():
    # model name
    model_name = 'unetv2'
    model_name = 'unet'

    # model sig
    model_sig = 'noise'

    # get model
    model = get_model(model_name, model_sig)


    # test data
    PATH_PIXELSHIFT = 'C:/Users/AI38/datasets/pixelshfit/PixelShift200_test'
    files = glob.glob(os.path.join(PATH_PIXELSHIFT, '*_3ch.npy'))


    patch_size = 128
    shape = np.load(files[0]).shape
    height, width, channels = np.load(files[0]).shape
    npatches_y, npatches_x = math.ceil(shape[0]/patch_size), math.ceil(shape[1]/patch_size)
    # print(arr_pred.shape)
    for idx, file in enumerate(files):
        arr = np.load(file)    # (0, 65535)
        arr = arr / (2**16 -1) # (0, 1)
        arr = arr ** (1/2.2)   # (0, 1)
        img_arr = Image.fromarray(  (arr*255).astype(np.uint8) )
        img_arr.save(os.path.join(PATH_PIXELSHIFT, f'inf_ref_%02d.png'%(idx+1)))

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
                sy = idx_y*patch_size
                ey = sy + patch_size
                sx = idx_x*patch_size
                ex = sx + patch_size

                if ey >= height:
                    ey = height-1
                    sy = height-patch_size-1

                if ex >= width:
                    ex = width-1
                    sx = width-patch_size-1

                arr_patch = arr[sy:ey, sx:ex,:]

                print(np.amin(arr_patch), np.amax(arr_patch) )

                # pre-process
                arr_patch = arr_patch**(1/2.2)
                arr_patch = (arr_patch*2) -1  # (0, 1) -> (-1, 1)

                # prediction
                pred = model.predict(arr_patch[np.newaxis,...])
                print(pred.shape)

                # post-process
                arr_pred[sy:ey, sx:ex, :] = (pred[0]+1)/2 #  (-1, 1) -> (0, 1)
                print(np.amin(arr_patch), np.amax(arr_patch), np.amin(arr_pred), np.amax(arr_pred))
                # exit()

        # arr_pred.astype(np.uint8)
        img_pred = Image.fromarray((arr_pred*255).astype(np.uint8))
        # name = os.path.join(PATH_PIXELSHIFT, f'inf_{model_name}_{model_sig}_%02d.png'%(idx+1))
        name = os.path.join(PATH_PIXELSHIFT, f'inf_{model_name}_{model_sig}_%02d_gamma.png'%(idx+1))
        img_pred.save(name)
        print(np.amin(img_pred), np.amax(img_pred), np.amin(arr_pred.astype(np.uint8)), np.amax(arr_pred.astype(np.uint8)))


        # exit()


if __name__ == '__main__':
    main()