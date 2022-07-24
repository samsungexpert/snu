import torch
import torchvision
import os
import numpy as np
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
from mymodel import mymodel
# from argument import get_args
import argparse

def train():
    ...

    model =  mymodel('resnet')

def main(args):
    print(args)

    # args
    model_name      = args.model_name
    dataset_name    = args.dataset_name
    dataset_path    = args.dataset_path
    input_size      = args.input_size
    device          = args.device

    print('model_name = ', model_name)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_name = ', dataset_name)
    print('dataset_path = ', dataset_path)


    # train()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='resnet', type=str,
                    choices=['resnet', 'unet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_name', default='sidd', type=str,
                    choices=['sidd','pixelshift'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path', default=os.path.join('dataset', 'sidd'), type=str,
                    choices=['sidd','pixelshift'],
                    help='(default=%(default)s)')
    argparser.add_argument('--device', default='cpu', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=128)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1e6)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch-size', type=int, help='mini batch size', default=128)
    args = argparser.parse_args()
    main(args)