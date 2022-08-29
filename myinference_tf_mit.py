



import os, glob
import numpy as np
import tensorflow as tf

from PIL import Image

import argparse

from tqdm import tqdm


def main(args):
    model_name = args.model_name
    model_sig  = args.model_sig
    base_dir  = args.base_dir


    ## find trining dataset
    src_dir = os.path.join(base_dir, 'images', 'train')
    folders = glob.glob(os.path.join(src_dir, '**', '**'))
    files = []
    folders.sort()
    for temp in folders:
        folder_num = temp.split('/')[-1]
        print(temp, folder_num)
        if int(folder_num) % 10 == 1:
            print(temp)
            files += glob.glob(os.path.join(temp, '**/**.png'), recursive=True)

    print(files[-1])
    print('len(files)', len(files))

    dirs = os.listdir(src_dir)
    for idx, d in enumerate(dirs):
        print(idx, d)

    # target folders
    new_name = 'images' + '_'  +  model_name + '_' + model_sig
    target_dir = os.path.join(base_dir,new_name, 'train')
    os.makedirs(target_dir, mode=777, exist_ok=True)

    ## model
    print(src_dir)
    print(target_dir)

    name_structure = os.path.join('model_dir', 'checkpoint', model_name + '_model_structure.h5')
    ckpt_path = os.path.join('model_dir', 'checkpoint',  model_name+'_'+model_sig )
    print(name_structure)
    print(ckpt_path)
    checkpoints = glob.glob(os.path.join(ckpt_path, '*.h5'))
    checkpoints.sort()


    model = tf.keras.models.load_model(name_structure)
    model.load_weights(checkpoints[-1])
    model.summary()
    print(checkpoints[-1])


    # inference & save
    ntot = len(files)
    for idx, f in tqdm(enumerate(files)):
        name = f.split('train')[-1]
        fsave = os.path.join(target_dir, name[0:])
        fsave = target_dir + name[:-4]
#        print('---->', name)
#        print('====>', fsave)
#        print('=-=->', target_dir)

        image = Image.open(f)
        arr = np.array(image)
#        print(arr.shape, np.amin(arr), np.amax(arr))

        # normalize
        arr = arr.astype(np.float32) / 255.
        arr = arr*2 -1

        # inference
        pred = model.predict(arr[np.newaxis,...]) # (-1, 1) --> (-1, 1)

        # expand (-1, 1) -> (0, 65535)
        pred = pred[0]
        pred = (pred +1) / 2 # (-1, 1) -> (0, 1)
        pred = pred * 65535
        pred = pred.astype(np.uint16)
#        print('>>>>> ',fsave[:-7])
#        print(pred.shape, np.amin(pred), np.amax(pred), pred.dtype)

        os.makedirs(fsave[:-7], mode=777, exist_ok=True)
        np.save(fsave, pred)


#        exit()
        #print(idx)

        pass


    print('done done')


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model_name',
            type=str,
            default='unetv2',
            help='resnet_flat, resnet_ed, bwunet, unet, unetv2')

    parser.add_argument(
            '--model_sig',
            type=str,
            default='noise',
            help='model postfix')

    parser.add_argument(
            '--base_dir',
            type=str,
            default='/data/team19/mit',
            help='mit data path')

    args = parser.parse_args()

    main(args)
