import tensorflow as tf
import numpy as np
import os
import glob
import random
from PIL import Image
# from tensorflow.keras.preprocessing import image



def write_tfrecords(file_list, idx_split, dataset_name, output_path):


    tfrecords_name = '%s_%03d.tfrecords' % (dataset_name, idx_split)
    tfrecords_name = os.path.join(output_path, tfrecords_name)

    if os.path.isfile(tfrecords_name):
        print('tfrecord %s exists' % tfrecords_name)
        return

    print('tfrecord not exists')

    writer = tf.io.TFRecordWriter(tfrecords_name)

    for index, file_name in enumerate(file_list):
        print('%d / %d' % (index+1, len(file_list)))

        image = Image.open(file_name)
        image_arr =  np.asarray(image).astype(np.uint8)
        image_byte = image_arr.tostring()

        example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_byte])
                    )
                })
        )


        writer.write(example.SerializeToString())
    writer.close()


def make_tfrecords(file_list, num_splits, dataset_name, output_path='.'):

    total_len = len(file_list)
    split_len = total_len // num_splits

    print(total_len, split_len)
    start_idx = np.array(list(range(num_splits)) )* split_len
    finish_idx= (np.array((list(range(num_splits)) )) + 1) * split_len

    print(start_idx)
    print(finish_idx)
    finish_idx[-1] = total_len

    # exit()
    # print(start_idx)
    # print(finish_idx)

    for idx_split in range(num_splits):
        idx_s = start_idx[idx_split]
        idx_f = finish_idx[idx_split]
        write_tfrecords(file_list[idx_s:idx_f], idx_split, dataset_name, output_path)


def main():

    num_of_splits = 20


    paths = ['/Users/bw/Datasets/mit/train']
    file_list=[]

    for path in paths:
        file_list += glob.glob(os.path.join(path, '**','*.png'), recursive=True)


    print(file_list[:5])
    random.shuffle(file_list)
    print(file_list[:5])

    print('# if files: ', len(file_list))
    exit()

    output_path = '//Users/bw/Datasets/mit'


    make_tfrecords(file_list, num_of_splits, dataset_name='mit10_1_train', output_path=output_path)

    # print(files)






    print('done')




if __name__ == '__main__':
    main()