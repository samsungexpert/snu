import os

import torch, torchvision, torchsummary
import numpy as np
from PIL import Image

from mymodel import mygen_model
from mydataset import *
from tqdm import tqdm

import argparse
import cv2

def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print('dev=', dev)

    print(args)
    # args
    dataset_path    = args.dataset_path
    model_name      = args.model_name
    model_type      = args.model_type # gan, cyclegan, generative
    model_sig       = args.model_sig
    dataset_name    = args.dataset_name
    input_size      = args.input_size
    device          = args.device
    inference_path  = args.inference_path
    if dev == 'cpu':
        device = torch.device('cpu')
    else:
        if device == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print('hello cuda')


    print('model_name = ', model_name)
    print('model_sig = ', model_sig)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_name = ', dataset_name)
    print('dataset_path = ', dataset_path)


    mytype = 'A'
    # mytype = 'C'
    base_path = os.path.join(dataset_path, dataset_name)
    valid_path  = os.path.join(base_path, 'test'+mytype)






    ckpt_path = os.path.join('checkpoint', model_type.lower(), dataset_name, model_name+model_sig)
    print('ckpt_path = ', ckpt_path)

    # ckpt_list = os.listdir(ckpt_path)
    print('CWD = ',  os.getcwd())
    ckpt_list = [s for s in os.listdir(ckpt_path) if s.endswith('.pth')]
    ckpt_list.sort()
    print(ckpt_list)

    if args.ckpt_step>0:
        ckpt_name = os.path.join(ckpt_path, '%s_%05d.pth'%(model_type, args.ckpt_step))
    else:
        ckpt_name = os.path.join(ckpt_path, ckpt_list[-1])


    checkpoint = torch.load(ckpt_name, map_location=device)

    # Get Model & load parameters
    model_G_rgb2raw = mygen_model(model_name).to(device)
    model_G_rgb2raw.load_state_dict(checkpoint['model_G_rgb2raw'])
    epoch = checkpoint["epoch"]


    print('use ckpt at epoch: ',  epoch)




    # Get dataset
    print('valid_path = ', valid_path)
    transform = give_me_transform('valid')
    dataloader = give_me_dataloader(SingleDataset(valid_path, transform, 16), args.batch_size)

    # inference gogo
    inference_path = os.path.join(inference_path,
                                f'{model_type}/{dataset_name}')
    os.makedirs(inference_path, exist_ok=True)

    model_G_rgb2raw.eval() # make model eval mode
    pbar = tqdm(dataloader)
    for idx, rgbimage in enumerate(pbar):
        if idx%4==0:
            pbar.set_description('Processing %s...  epoch %d' % (idx, epoch))

        # get real rgb images
        real_rgb = rgbimage.to(device)

        # get fake raw images
        real_raw = degamma(real_rgb, device)

        # get fake raw
        fake_raw = model_G_rgb2raw(real_rgb)


        # print(real_rgb.shape, real_raw.shape, fake_raw.shape)
        # exit()
        f = lambda x : x.permute(0,2,3,1).detach().numpy()
        real_rgb = f(real_rgb)
        real_raw = f(real_raw)
        fake_raw = f(fake_raw)

        print(fake_raw.shape)
        for i in range(fake_raw.shape[0]):
            print('---> ', i)

            # real rgb
            rgb = real_rgb[i]

            # real raw
            rraw = real_raw[i]

            # fake raw
            fraw = fake_raw[i]

            # diff raw
            diff = np.abs(rraw - fraw)


            real = np.concatenate((rgb, rraw),axis=1)
            fake_and_diff = np.concatenate((fraw, diff), axis=1)
            gdn = np.concatenate((real, fake_and_diff), axis=0)


            # gdn = np.concatenate((rgb, img), axis=1)

            gdn = (gdn+1)/2

            gdn = (gdn*255).astype(np.uint8)

            print(np.amin(gdn), np.amax(gdn), 'gdn.shape ', gdn.shape)


            img = Image.fromarray(gdn)
            name = os.path.join(inference_path,
                   f'inf_{model_type}_%02d.png'%(idx*args.batch_size+i))
            img.save(name)
        # exit()


    print('done done')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='bwunet', type=str,
                    choices=['resnet', 'unet', 'bwunet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--model_type', default="gan", type=str,
                    choices=['gan', 'cyclegan', 'generative'], help='(default=gan, training type, GAN, CycleGAN, Generative')
    argparser.add_argument('--dataset_name', default='apple2orange', type=str,
                    choices=['sidd', 'pixelshift', 'apple2orange'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path', default=os.path.join('datasets'), type=str,
                    help='(default=datasets)')
    argparser.add_argument('--inference_path', default=os.path.join('inference'), type=str,
                    help='(default=inference, inference output path')
    argparser.add_argument('--model_sig', default="",
                    type=str, help='(default=model signature for same momdel different ckpt/log path)')

    argparser.add_argument('--input_size', type=int, help='input size', default=256)
    argparser.add_argument('--ckpt_step', type=int, help='ckpt step #', default=-1)

    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)

    argparser.add_argument('--device', default='cuda', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    args = argparser.parse_args()
    main(args)