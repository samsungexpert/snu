import random
#import tensorflow as tf
import numpy as np
import os
# import scipy, scipy.io
import glob


def make_3ch(base_path, name_4ch, name_3ch):
    path_4ch = os.path.join(base_path, name_4ch)
    path_3ch = os.path.join(base_path, name_3ch)
    os.makedirs(path_3ch, exist_ok=True)

    files_4ch = glob.glob(os.path.join(path_4ch, '*.npy'))
    aa = np.load(files_4ch[0])
    H, W, C = aa.shape

    print('len files4ch ', len(files_4ch))

    p_size = 128
    img_patch_3ch = np.zeros((p_size, p_size, 3))
    for idx, filename in enumerate(files_4ch):
        if idx % 10 == 0:
            print(idx, ' / ', len(files_4ch))

        img_patch = np.load(filename)
        img_name = filename.split("/")[-1].split('.')[0]

        save_name = os.path.join(path_3ch, img_name + '_3ch.npy')
        img_patch_3ch = img_patch[..., [0, 1, 3]]

        with open(save_name, 'wb') as f:
            np.save(f, img_patch_3ch.astype(np.uint16))


#def write_tfrecords(file_list, idx_split, dataset_name, output_path):
#
#    tfrecords_name = '%s_%03d.tfrecords' % (dataset_name, idx_split)
#    tfrecords_name = os.path.join(output_path, tfrecords_name)
#
#    if os.path.isfile(tfrecords_name):
#        print('OHNO -->. tfrecord (%s) exists' % tfrecords_name)
#        #return
#
#    print('tfrecord not exists')
#    print('--->', tfrecords_name)
#
#    file_list.sort()
#    writer = tf.io.TFRecordWriter(tfrecords_name)
#
#    for index, file_name in enumerate(file_list):
#        print('%d / %d' % (index + 1, len(file_list)), file_name)
#
#        image_arr = np.load(file_name)
#        image_byte = image_arr.tobytes()
#
#        example = tf.train.Example(features=tf.train.Features(
#            feature={
#                'image':
#                tf.train.Feature(bytes_list=tf.train.BytesList(
#                    value=[image_byte]))
#            }))
#
#        writer.write(example.SerializeToString())
#    writer.close()


#def make_tfrecords(paths, num_splits, dataset_name, output_path='.'):
#
#    print(paths)
#    print(dataset_name)
#    file_list = []
#    for path in paths:
#        p = os.path.join(path, '*.npy')
#        print('-----> ', p)
#        file_list += glob.glob(os.path.join(path, '*.npy'), recursive=True)
#
#    assert len(file_list) > 0, f'no file is found in ({paths})'
#    random.shuffle(file_list)
#    random.shuffle(file_list)
#
#    total_len = len(file_list)
#    split_len = total_len // num_splits
#
#    print(total_len, split_len)
#    start_idx = np.array(list(range(num_splits))) * split_len
#    finish_idx = (np.array((list(range(num_splits)))) + 1) * split_len
#
#    print(start_idx)
#    print(finish_idx)
#    finish_idx[-1] = total_len
#
#    # exit()
#    # print(start_idx)
#    # print(finish_idx)
#
#    for idx_split in range(num_splits):
#        print(f'... {idx_split+1} / {num_splits}')
#        idx_s = start_idx[idx_split]
#        idx_f = finish_idx[idx_split]
#        write_tfrecords(file_list[idx_s:idx_f], idx_split, dataset_name,
#                        output_path)
#
#    print('done make_tfrecords')


def main():
    base_path = '/data03/team01/pixelshift'
    train_dir = 'trainC'
    valid_dir = 'validC'
    test_dir  = 'testC'
    viz_dir   = 'vizC'
    
    train_3ch_dir = train_dir + '_3ch'
    valid_3ch_dir = valid_dir + '_3ch'
    test_3ch_dir  = test_dir + '_3ch'
    viz_3ch_dir   = viz_dir + '_3ch'
    
    train_path = os.path.join(base_path, train_dir)
    valid_path = os.path.join(base_path, valid_dir)
    test_path  = os.path.join(base_path, test_dir)
    viz_path   = os.path.join(base_path, viz_dir)
    
    train_3ch_path = os.path.join(base_path, train_3ch_dir)
    valid_3ch_path = os.path.join(base_path, valid_3ch_dir)
    test_3ch_path  = os.path.join(base_path, test_3ch_dir)
    viz_3ch_path   = os.path.join(base_path, viz_3ch_dir)

    lst_train = os.listdir(train_path)
    lst_valid = os.listdir(valid_path)
    lst_test  = os.listdir(test_path)
    lst_viz   = os.listdir(viz_path)
    print('test: ', len(lst_train))
    print('valid: ', len(valid_path))
    print('test: ', len(test_path))
    print('viz: ', len(viz_path))
    
    
    # make 3ch npy
    make_3ch(base_path, train_dir, train_dir+'_3ch')
    make_3ch(base_path, valid_dir, valid_dir+'_3ch')
    make_3ch(base_path, test_dir, test_dir+'_3ch')
    make_3ch(base_path, viz_dir, viz_dir+'_3ch')
    #    exit()

    ## make tfrecords
    # num_of_splits = 20
    # tfrecords_path = os.path.join(base_path, 'tfrecords')
    #make_tfrecords([train_3ch_path], num_of_splits, dataset_name='pixelshift_train', output_path=tfrecords_path)
    #make_tfrecords([valid_3ch_path], num_splits=4, dataset_name='pixelshift_valid', output_path=tfrecords_path)
    # make_tfrecords([viz_3ch_path],
    #                num_splits=1,
    #                dataset_name='pixelshift_viz10',
    #                output_path=tfrecords_path)

    print('done main')


if __name__ == '__main__':
    main()
