import os
import cv2
import glob
import mat73
import numpy as np
import scipy.io
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if False:

    base_dir = 'PixelShift200_train'
    print(base_dir)
    mat_list = glob.glob(os.path.join(base_dir, '**/*.mat'), recursive=True)
    for f in reversed(mat_list):
        print(f)

        mymat = scipy.io.loadmat(f)
        print(mymat.keys())
        for i in range(3):
            k = list(mymat.keys())[i]
            v = mymat[k]
            print(i, v)

        image = mymat[list(mymat.keys())[-1]]
        image = np.array(image)
        name = f.split('.mat')[0]
        num = int(name.split('_')[-1])

        ridx = name.rfind('_')
        new_name = name[:ridx]
        new_name = new_name + '_' + '%03d'%num


        if 1:#not os.path.isfile(new_name + '.npy'):
            np.save(new_name, image)
            print('-----> save: ', new_name + '.npy')
        else:
            print('-----> skip')

else:
    base_dir = 'PixelShift200_train'

    npy_list = glob.glob(os.path.join(base_dir, '**/*.npy'), recursive=True)
    npy_list

    BITS = 12
    MAX_VAL = 2**BITS -1

    for f in reversed(npy_list):
        print(f)
        arr = np.load(f)
        print(arr.shape, arr.dtype)


        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint16)
        rgb[...,0] = arr[...,0]
        rgb[...,1] = arr[...,2]
        rgb[...,2] = arr[...,-1]

    #     rgb/=MAX_VAL
        print(type(rgb[0,0,0]))
        print(np.amin(rgb[...,0]), np.amax(rgb[...,0]))
        print(np.amin(rgb[...,1]), np.amax(rgb[...,1]))
        print(np.amin(rgb[...,2]), np.amax(rgb[...,2]))


        pngname = f.split('.npy')[0] + '_%d'%BITS+'.png'
        if not os.path.isfile(pngname):
            io.imsave(pngname, rgb/MAX_VAL)