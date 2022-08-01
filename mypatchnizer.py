

from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def get_files(path):
    # pattern = os.path.join(path, '*.png')
    pattern = os.path.join(path, '*.npy')
    files = glob.glob(pattern, recursive=True)
    return files

def get_image_list_from_paths(*args):

    image_path = ''
    for path in args:
       image_path = os.path.join(image_path, path)

    print('image_path = ', image_path)

    images_hr_train = get_files(image_path)
    images_hr_train.sort()

    return images_hr_train

def read_image_from_path(image_path):
    image = Image.open(image_path)
    image_arr = np.array(image)
    print(image_path, image_arr.shape)


def crop_batch(file_name, image_arr, crop_size=128):

    prefix = file_name.split('.')[0]

    # idx_del = file_name.rfind('\\')
    idx_del = file_name.rfind('/')

    ext = file_name[-4:]
    path = file_name[:idx_del]
    file_name = file_name[idx_del+1:-4]
    print('path=', path, ', file_name=', file_name)
    file_name_patch = os.path.join(path, 'patches')
    os.makedirs(file_name_patch, exist_ok=True)

    height, width , _ = image_arr.shape

    width_leftover = width % crop_size
    height_levtover = height % crop_size

    n_patch_width = width // crop_size
    n_patch_height = height // crop_size

    print(image_arr.shape, height, width, height_levtover, width_leftover, n_patch_height, n_patch_width)


    idx = 1
    for hidx in range(n_patch_height):
        for widx in range(n_patch_width):
            idx_h_start = hidx * crop_size + height_levtover//2
            idx_h_end = idx_h_start + crop_size

            idx_w_start = widx * crop_size + width_leftover//2
            idx_w_end = idx_w_start + crop_size

            image_arr_crop = image_arr[idx_h_start:idx_h_end, idx_w_start:idx_w_end]
            print(image_arr_crop.shape)





            patch_name = file_name + ('_%03d'% idx )+ ext

            file_name_patch = os.path.join(path, 'patches')
            # os.makedirs(file_name_patch, exist_ok=True)
            # print('file_name_patch, ', file_name_patch)
            file_name_patch = os.path.join(file_name_patch, patch_name )
            print('file_name_patch, ', file_name_patch)

                # exit()
            if '.npy' in ext:
                np.save(file_name_patch, image_arr_crop)

            else:
                image_crop = Image.fromarray(image_arr_crop)
                image_crop.save(file_name_patch, 'png')


            idx += 1

    print('width ', width, ', height', height)


def crop_batch_npy(file_name, image_arr, crop_size=128):

    prefix = file_name.split('.')[0]

    # idx_del = file_name.rfind('\\')
    idx_del = file_name.rfind('/')

    path = file_name[:idx_del]
    file_name = file_name[idx_del+1:-4]
    print('path=', path, ', file_name=', file_name)

    height, width , _ = image_arr.shape

    width_leftover = width % crop_size
    height_levtover = height % crop_size

    n_patch_width = width // crop_size
    n_patch_height = height // crop_size

    print(image_arr.shape, height, width, height_levtover, width_leftover, n_patch_height, n_patch_width)


    idx = 1
    for hidx in range(n_patch_height):
        for widx in range(n_patch_width):
            idx_h_start = hidx * crop_size + height_levtover//2
            idx_h_end = idx_h_start + crop_size

            idx_w_start = widx * crop_size + width_leftover//2
            idx_w_end = idx_w_start + crop_size

            image_arr_crop = image_arr[idx_h_start:idx_h_end, idx_w_start:idx_w_end]
            print(image_arr_crop.shape)


            patch_name = file_name + ('_%03d'% idx )+ '.png'

            file_name_patch = os.path.join(path, 'patches')
            # print('file_name_patch, ', file_name_patch)
            file_name_patch = os.path.join(file_name_patch, patch_name )
            print('file_name_patch, ', file_name_patch)

            # exit()

            image_crop = Image.fromarray(image_arr_crop)
            image_crop.save(file_name_patch, 'png')


            idx += 1

    print('width ', width, ', height', height)




def crop_images_from_path(paths, recursive=True, crop_size=128):

    if isinstance(paths, str):
        image_path = paths
    else:
        image_path = ''
        for path in paths:
            image_path = os.path.join(image_path, path)

    patch_path = os.path.join(image_path, 'patches')
    print('patch_path = ', patch_path)
    # exit()
    os.makedirs(patch_path, exist_ok=True)

    file_names = get_image_list_from_paths(image_path)

    for file_name in file_names:
        if '.npy' in file_name[-4:]:
            ...
            image_arr = np.load(file_name)
        else:
            image = Image.open(file_name)
            image_arr = np.array(image)
        crop_batch(file_name, image_arr)


    # print(file_names)



def main():

    div2k_path = 'E:\dataset\div2k'
    div2k_path = '/dataset/DIV2K/imgs'



    hr_train_path = 'DIV2K_train_HR'
    hr_valid_path = 'DIV2K_valid_HR'


    pixelshift_path_valid = '/Users/bw/gdrive/snu/datasets/pixelshift/valid'

    # crop_images_from_path((div2k_path, hr_train_path))
    # crop_images_from_path((div2k_path, hr_valid_path))

    crop_images_from_path((pixelshift_path_valid))

   # list_hr_train = get_image_list_from_paths(div2k_path, hr_train_path)
   # for file in list_hr_train:
   #     image = Image.open(file)
   #     image_arr = np.array(image)
   #     print(image_arr.shape)

   #     exit()

    pass


if __name__ == '__main__':

    main()



    print('write done')
