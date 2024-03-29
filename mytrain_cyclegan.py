# https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo

import os, shutil
import time
import argparse
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch import nn, optim

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
from mymodel import mygen_model, mydisc_model, set_requires_grad
from myutils import init_weight, ImagePool, LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

from mydataset import * #give_me_dataloader, SingleDataset, give_me_transform, give_me_test_images, give_me_comparison, degamma
from myloss import BayerLoss
import matplotlib.pyplot as plt


def degamma_visualization(rgbimage):
    rgb1 = rgbimage[0]
    rgb1 = rgb1.permute(1,2,0)
    rgb2 = (((rgb1+1)/2)**(2.2))*2 - 1
    rgb3 = (((rgb1+1)/2)**(0.45))*2 - 1


    plt.subplot(1, 3, 1)
    plt.imshow(rgb1)
    plt.title('rgb1 - before gamma')

    plt.subplot(1, 3, 2)
    plt.imshow(rgb2)
    plt.title('rgb2 - after gamma')

    plt.subplot(1, 3, 3)
    plt.imshow(rgb3)
    plt.title('rgb3 - after degamma')
    plt.show()

def identity(x):
    return x


def train(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('dev=', dev)

    print(args)
    # args
    model_name      = args.model_name
    dataset_name    = args.dataset_name
    dataset_path    = args.dataset_path
    input_size      = args.input_size
    batch_size      = args.batch_size
    device          = args.device
    if dev == 'cpu':
        device = torch.device('cpu')
    else:
        if device == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print('hello cuda')


    print('model_name = ', model_name)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_name = ', dataset_name)
    print('dataset_path = ', dataset_path)

    try:
        shutil.rmtree('runs')
        shutil.rmtree('checkpoint')
    except:
        pass

    # dataset
    base_path = os.path.join(dataset_path, dataset_name)
    print('base_path: ', base_path)

    # path
    mytype = 'A'
    train_path = os.path.join(base_path, 'train'+mytype)
    valid_path  = os.path.join(base_path, 'test'+mytype)
    test_path = os.path.join('imgs', dataset_name)
    mydata_path = {'train': train_path,
                   'valid': valid_path,
                   'test' : test_path}


    # transform
    transform = {'train': give_me_transform('train'),
                 'valid': give_me_transform('valid'),
                 'test': give_me_transform('test')}

    # dataloader


    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train']), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid']), batch_size),
                  'test' : give_me_dataloader(SingleDataset(mydata_path['test'],  transform['test']),  batch_size=2) }


    nsteps={}
    for state in ['train', 'valid']:
        nsteps[state] = len(dataloader[state])
        print('len(%s): '%state, len(dataloader[state]))

    # model
    # model_G_rgb2raw = mygen_model('resnet').to(device)
    # model_G_raw2rgb = mygen_model('resnet').to(device)
    model_G_rgb2raw = mygen_model('bwunet').to(device)
    model_G_raw2rgb = mygen_model('bwunet').to(device)
    model_D_rgb     = mydisc_model('basic').to(device)
    model_D_raw     = mydisc_model('basic').to(device)


    ## ckpt save load if any
    ckpt_path_name = f"checkpoint/{dataset_name}"
    os.makedirs(ckpt_path_name, exist_ok=True)
    ckpt_list = os.listdir(args.checkpoint_path)
    if (args.checkpoint_path is not None) and \
        (len(ckpt_list) > 0) :

        ckpt_list.sort()
        ckpt_name = os.path.join(ckpt_path_name, ckpt_list[-1])
        print('ckpt name = ', ckpt_name)
        checkpoint = torch.load(ckpt_name, map_location=device)

        model_G_rgb2raw.load_state_dict(checkpoint["model_G_rgb2raw"])
        model_G_raw2rgb.load_state_dict(checkpoint["model_G_raw2rgb"])
        model_D_rgb.load_state_dict(checkpoint["model_D_rgb"])
        model_D_raw.load_state_dict(checkpoint["model_D_raw"])
        epoch = checkpoint["epoch"]
    else:
        epoch = 0
        try:
            fname = os.path.join(ckpt_path_name, f"{epoch}.pth")
            if os.path.exists(fname):
                fname = fname.split('.pth')[0] + '_1.pth'
            torch.save(
                    {
                        "model_G_rgb2raw": model_G_rgb2raw.state_dict(),
                        "model_G_raw2rgb": model_G_raw2rgb.state_dict(),
                        "model_D_rgb": model_D_rgb.state_dict(),
                        "model_D_raw": model_D_raw.state_dict(),
                        "epoch": epoch,
                    },
                    fname,
            )
        except:
            print('skip')



    # visualize test images
    test_batch = next( iter(dataloader['test']))

    summary = SummaryWriter()
    test_images = give_me_visualization(model_G_rgb2raw, model_G_raw2rgb, 'cpu', test_batch)
    summary.add_image('Generated_pairs', test_images.permute(2,0,1), 0)
    # plt.imshow(test_images)
    # plt.show()
    # exit()


    # make model in training mode
    model_G_rgb2raw.train()
    model_G_raw2rgb.train()
    model_D_rgb.train()
    model_D_raw.train()

    # Loss
    criterion_bayer = BayerLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_GAN = nn.MSELoss()



    # Optimizer, Schedular
    optim_G = optim.Adam(
        list(model_G_rgb2raw.parameters()) + list(model_G_raw2rgb.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optim_D_raw = optim.Adam(model_D_rgb.parameters(), lr=args.lr)
    optim_D_rgb = optim.Adam(model_D_raw.parameters(), lr=args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)
    scheduler_D_rgb2raw = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_raw, lr_lambda=lr_lambda
    )
    scheduler_D_raw2rgb = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_rgb, lr_lambda=lr_lambda
    )

    # Training
    # logger for tensorboard.

    disp_train = LossDisplayer(["G_train",
                                "G_GAN_train",
                                "G_Identity_train",
                                "G_Cycle_train",
                                "D_train"])
    disp_valid = LossDisplayer(["G_valid",
                                "G_GAN_valid",
                                "G_Identity_valid",
                                "G_Cycle_valid",
                                "D_valid"])
    disp = {'train':disp_train, 'valid':disp_valid}

    step = 0
    loss_best_G = 1e10

    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")


        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, rgbimage in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))

                if state == 'train' and idx == 0:
                    # train mode
                    model_G_rgb2raw.train()
                    model_G_raw2rgb.train()
                    optim_G.zero_grad()
                elif state == 'valid' and idx == 0:
                    # eval mode
                    model_G_rgb2raw.eval()
                    model_G_raw2rgb.eval()


                # degamma to make raw image
                real_rgb = rgbimage.to(device)
                real_raw = degamma(rgbimage, device).to(device)


                # -----------------
                # Train Generators
                # -----------------



                #### Forward
                fake_raw = model_G_rgb2raw(real_rgb)
                fake_rgb = model_G_raw2rgb(real_raw)
                rec_rgb = model_G_raw2rgb(fake_raw)
                rec_raw = model_G_rgb2raw(fake_rgb)
                # rec_rgb = rec_rgb.to(device)
                # rec_raw = rec_raw.to(device)


                ## freeze D when training Generators
                set_requires_grad([model_D_rgb, model_D_raw], False)


                if state == 'train':
                    optim_G.zero_grad()


                loss_G = 0

                # Identity Loss
                if True: #args.identity:
                    identity_raw = model_G_raw2rgb(real_rgb)
                    identity_rgb = model_G_rgb2raw(real_raw)

                    loss_identity_rgb = criterion_identity(identity_rgb, real_raw)
                    loss_identity_raw = criterion_identity(identity_raw, real_rgb)
                    loss_identity = 0.5 * (loss_identity_rgb + loss_identity_raw)

                    loss_G += args.lambda_ide *  loss_identity


                # GAN loss
                pred_fake_raw = model_D_raw(fake_raw)
                pred_fake_rgb = model_D_rgb(fake_rgb)

                loss_GAN_raw  = criterion_GAN(pred_fake_raw, torch.ones_like(pred_fake_raw))
                loss_GAN_rgb  = criterion_GAN(pred_fake_rgb, torch.ones_like(pred_fake_rgb))
                loss_GAN = 0.5 * ( loss_GAN_raw + loss_GAN_rgb )

                loss_G += loss_GAN

                # Cycle Loss
                loss_cycle_rgb = criterion_cycle(rec_rgb, real_rgb)
                loss_cycle_raw = criterion_cycle(real_raw, real_raw)
                loss_cycle = 0.5 * (loss_cycle_rgb + loss_cycle_raw)
                loss_G += args.lambda_cycle * loss_cycle


                if state == 'train':
                    loss_G.backward()
                    optim_G.step()



                # -----------------
                # Discriminator Setup
                # -----------------
                if state == 'train':
                    optim_D_raw.zero_grad()
                    optim_D_rgb.zero_grad()
                    require_grad_for_D = True
                else:
                    require_grad_for_D = False

                 ## freeze D when training Generators
                set_requires_grad([model_D_rgb, model_D_raw], require_grad_for_D)



                # -----------------
                # Train Discriminator Raw
                # -----------------
                pred_real_raw = model_D_raw(real_raw)
                pred_fake_raw = model_D_raw(fake_raw.detach())

                loss_D_raw_real = criterion_GAN(pred_real_raw, torch.ones_like( pred_real_raw))
                loss_D_raw_fake = criterion_GAN(pred_fake_raw, torch.zeros_like(pred_fake_raw))
                loss_D_raw = 0.5 * ( loss_D_raw_real + loss_D_raw_fake )


                # -----------------
                # Train Discriminator RGB
                # -----------------
                pred_real_rgb = model_D_rgb(real_rgb)
                pred_fake_rgb = model_D_rgb(fake_rgb.detach())


                loss_D_rgb_real = criterion_GAN(pred_real_rgb, torch.ones_like( pred_real_rgb))
                loss_D_rgb_fake = criterion_GAN(pred_fake_rgb, torch.zeros_like(pred_fake_rgb))
                loss_D_rgb = 0.5 * ( loss_D_rgb_real + loss_D_rgb_fake )

                loss_D = 0.5 * (loss_D_raw + loss_D_rgb)



                # backward for discriminator
                if state == 'train':
                    loss_D_raw.backward()
                    loss_D_rgb.backward()
                    optim_D_raw.step()
                    optim_D_rgb.step()
                    step+=1
                    if loss_best_G > loss_G:
                        loss_best_G = loss_G
                        summary.add_scalar(f"loss_best_G{state}", loss_best_G, step)



                # -----------------
                # record loss for tensorboard
                # -----------------
                disp[state].record([loss_G, loss_GAN, loss_identity, loss_cycle, loss_D])
                if step%20==0 :
                    avg_losses = disp[state].get_avg_losses()
                    summary.add_scalar(f"loss_G_{state}",           avg_losses[0], step)
                    summary.add_scalar(f"loss_G_G_GAN_{state}",     avg_losses[1], step)
                    summary.add_scalar(f"loss_G_Identity_{state}",  avg_losses[2], step)
                    summary.add_scalar(f"loss_G_Cycle_{state}",     avg_losses[3], step)
                    summary.add_scalar(f"loss_D_{state}",           avg_losses[4], step)

                    print(f'{state} : epoch{epoch}, step{step}------------------------------------------------------')
                    print('loss_G:          \t', avg_losses[0])
                    print('loss_G_G_GAN:    \t', avg_losses[1])
                    print('loss_G_Identity: \t', avg_losses[2])
                    print('loss_G_Cycle:    \t', avg_losses[3])
                    print('loss_D:          \t', avg_losses[4])

                    disp[state].reset()

                    # log images
                    test_images = give_me_visualization(model_G_rgb2raw, model_G_raw2rgb, device, test_batch)
                    summary.add_image('Generated_pairs', test_images.permute(2,0,1), step)


        # Step scheduler
        scheduler_G.step()
        scheduler_D_rgb2raw.step()
        scheduler_D_raw2rgb.step()


        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "model_G_rgb2raw": model_G_rgb2raw.state_dict(),
                    "model_G_raw2rgb": model_G_raw2rgb.state_dict(),
                    "model_D_rgb2raw": model_D_rgb.state_dict(),
                    "model_D_raw2rgb": model_D_raw.state_dict(),
                    "epoch": epoch,
                },
                os.path.join("checkpoint", dataset_name, "cycle_gan_%05d.pth"%epoch),
            )


    print('done done')



def main(args):
    train(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='resnet', type=str,
                    choices=['resnet', 'unet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_name', default='apple2orange', type=str,
                    choices=['sidd','pixelshift', 'apple2orange'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path', default=os.path.join('datasets'), type=str,
                    choices=['sidd','pixelshift', 'apple2orange'],
                    help='(default=%(default)s)')
    argparser.add_argument('--checkpoint_path', default=f"checkpoint/apple2orange",
                    type=str, help='(default=%(default)s)')

    argparser.add_argument('--device', default='cuda', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=256)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)
    argparser.add_argument("--lambda_ide", type=float, default=10)
    argparser.add_argument("--lambda_cycle", type=float, default=5)
    argparser.add_argument("--identity", action="store_true")
    args = argparser.parse_args()
    main(args)