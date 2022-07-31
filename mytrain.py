import os
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
from mymodel import mygen_model, mydisc_model
from myutils import init_weight, ImagePool, LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

from mydataset import give_me_dataloader, SingleDataset, give_me_transform
from myloss import BayerLoss, degamma
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



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('train with', device)
    print(args)
    # args
    model_name      = args.model_name
    dataset_name    = args.dataset_name
    dataset_path    = args.dataset_path
    input_size      = args.input_size
    batch_size      = args.batch_size
    device          = args.device
    learning_rate   = args.lr

    print('model_name = ', model_name)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_name = ', dataset_name)
    print('dataset_path = ', dataset_path)

    # dataset
    base_path = os.path.join(dataset_path, dataset_name)
    print('base_path: ', base_path)

    # path
    mytype = 'A'
    train_path = os.path.join(base_path, 'train'+mytype)
    valid_path  = os.path.join(base_path, 'test'+mytype)
    mydata_path = {'train': train_path,
                   'valid': valid_path}

    # transform
    transform = {'train': give_me_transform('train'),
                 'valid': give_me_transform('valid')}

    # dataloader
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train']), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid']),  batch_size) }

    nsteps={}
    for state in ['train', 'valid']:
        nsteps[state] = len(dataloader[state])
        print('len(%s): '%state, len(dataloader[state]))

    # model
    model_G_rgb2raw = mygen_model('resnet').to(device)
    model_G_raw2rgb = mygen_model('resnet').to(device)
    model_D_rgb2raw = mydisc_model('basic').to(device)
    model_D_raw2rgb = mydisc_model('basic').to(device)


    ## ckpt save load if any
    if (args.checkpoint_path is not None) and \
        (len(os.listdir(args.checkpoint_path) ) > 0) :
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model_G_rgb2raw.load_state_dict(checkpoint["model_G_rgb2raw_state_dict"])
        model_G_raw2rgb.load_state_dict(checkpoint["model_G_raw2rgb_state_dict"])
        model_D_rgb2raw.load_state_dict(checkpoint["model_D_rgb2raw_state_dict"])
        model_D_raw2rgb.load_state_dict(checkpoint["model_D_raw2rgb_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        os.makedirs(f"checkpoint/{dataset_name}", exist_ok=True)
        epoch = 0

    model_G_rgb2raw.train()
    model_G_raw2rgb.train()
    model_D_rgb2raw.train()
    model_D_raw2rgb.train()

    # Loss
    criterion_bayer = BayerLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_GAN = nn.MSELoss()

    dispG = LossDisplayer(["G_GAN", "G_recon", "Dg"])
    dispF = LossDisplayer(["F_GAN", "F_recon", "Df"])
    summary = SummaryWriter()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        list(model_G_rgb2raw.parameters()) + list(model_G_raw2rgb.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optim_D_raw = optim.Adam(model_D_rgb2raw.parameters(), lr=args.lr)
    optim_D_rgb = optim.Adam(model_D_raw2rgb.parameters(), lr=args.lr)

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
    logger = SummaryWriter()
    step = 0
    loss_G_RGB2Raw  = 0
    loss_D_RGB2Raw  = 0
    loss_G_Raw2RGB  = 0
    loss_D_Raw2RGB  = 0


    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")


        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, rgbimage in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))


                # degamma to make raw image
                real_rgb     = rgbimage.to(device)
                real_raw = degamma(rgbimage).to(device)

                # Forward
                fake_raw = model_G_rgb2raw(real_rgb)
                fake_rgb = model_G_raw2rgb(real_raw)
                cycle_rgb = model_G_raw2rgb(fake_raw)
                cycle_raw = model_G_rgb2raw(fake_rgb)

                pred_fake_raw = model_D_rgb2raw(fake_raw)
                pred_fake_rgb = model_D_raw2rgb(fake_rgb)


                # Calculate and backward generator model losses
                loss_cycle_A = criterion_cycle(cycle_rgb, real_rgb)
                loss_cycle_B = criterion_cycle(cycle_raw, real_raw)

                loss_GAN_raw = criterion_GAN(pred_fake_raw, torch.ones_like(pred_fake_raw))
                loss_GAN_rgb = criterion_GAN(pred_fake_rgb, torch.ones_like(pred_fake_rgb))

                loss_G = (
                    args.lambda_ide * (loss_cycle_A + loss_cycle_B)
                    + loss_GAN_raw + loss_GAN_rgb
                )

                if args.identity:
                    identity_rgb = model_G_raw2rgb(real_rgb)
                    identity_raw = model_G_rgb2raw(real_raw)
                    loss_identity_rgb = criterion_identity(identity_rgb, real_rgb)
                    loss_identity_raw = criterion_identity(identity_raw, real_raw)
                    loss_G += 0.5 * args.lambda_ide * (loss_identity_rgb + loss_identity_raw)




                # calculate loss
                # loss_bayer = criterion_bayer(real_raw, network_g_raw_output)
                # loss_gan = criterion_GAN(real_raw, fake_raw_output)


                if state == 'train':
                    # train mode



                    # Backward G
                    optim_G.zero_grad()
                    loss_G.backward()
                    optim_G.step()
                    loss_g_step = loss_G.item()


                    # Calculate and backward discriminator model losses
                    # Backward D_raw
                    pred_real_raw = model_D_rgb2raw(real_raw)


                    loss_D_raw = 0.5 * ( criterion_GAN(pred_real_raw, torch.ones_like(pred_real_raw))
                                       + criterion_GAN(pred_fake_raw, torch.zeros_like(pred_fake_raw)) )

                    optim_D_raw.zero_grad()
                    loss_D_raw.backward(retain_graph=True)
                    optim_D_raw.step()

                    # Backward D_rgb
                    pred_real_rgb = model_D_rgb2raw(real_rgb)

                    loss_D_rgb = 0.5 * ( criterion_GAN(pred_real_rgb, torch.ones_like(pred_real_rgb))
                                       + criterion_GAN(pred_fake_rgb, torch.zeros_like(pred_fake_rgb)) )

                    optim_D_rgb.zero_grad()
                    loss_D_rgb.backward(retain_graph=True)
                    optim_D_rgb.step()


                    step+=1






                # if state == 'valid':
                #     pass


                # if step %1 == 0:
                #     logger.add_scalar('loss_G', loss_g_rgb2raw, step)

        # Step scheduler
        scheduler_G.step()
        scheduler_D_rgb2raw.step()
        scheduler_D_raw2rgb.step()

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
                    help='(default=datasets)')
    argparser.add_argument('--checkpoint_path', default=f"checkpoint/apple2orange",
                    type=str, help='(default=%(default)s)')
    argparser.add_argument('--device', default='cpu', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=256)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)
    argparser.add_argument("--lambda_ide", type=float, default=10)
    argparser.add_argument("--identity", action="store_true")
    args = argparser.parse_args()
    main(args)