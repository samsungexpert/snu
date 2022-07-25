import os
import time
import argparse
import torch
import torchvision
import numpy as np
from torch import nn, optim
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
from mymodel import mymodel
from myutils import init_weight, ImagePool, LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

from mydataset import give_me_dataloader, SingleDataset, give_me_transform

def train(args):
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
    test_path  = os.path.join(base_path, 'test'+mytype)
    mydata_path = {'train' : train_path,
                   'test'  : test_path}

    # transform
    transform = {'train': give_me_transform('train'),
                 'test' : give_me_transform('test')}

    # dataloader
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train']), batch_size),
                  'test':  give_me_dataloader(SingleDataset(mydata_path['test'],  transform['test']),  batch_size) }

    # model
    model_g = mymodel('resnet', input_size=(input_size, input_size))
    model_d = 1


    #####################################
    ######
    # Loss
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_GAN = nn.MSELoss()

    disp = LossDisplayer(["G_GAN", "G_recon", "D"])
    summary = SummaryWriter()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        model_g.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optim_D = optim.Adam(netD_A.parameters(), lr=args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    # scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)
    # scheduler_D_A = optim.lr_scheduler.LambdaLR(
    #     optimizer=optim_D_A, lr_lambda=lr_lambda
    # )
    # scheduler_D_B = optim.lr_scheduler.LambdaLR(
    #     optimizer=optim_D_B, lr_lambda=lr_lambda
    # )

    #####
    #####################################



    # Training
    os.makedirs(f"checkpoint/{dataset_name}", exist_ok=True)
    epoch=0
    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")
        break





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
    argparser.add_argument('--device', default='cpu', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=256)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1e6)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=128)
    args = argparser.parse_args()
    main(args)