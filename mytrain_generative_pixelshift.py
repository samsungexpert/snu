# https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo

import os, shutil
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim

# CUDA_VISIBLE_DEVICES=5 python mytrain_generative_pixelshift.py --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=bwunet --model_sig=wine
# CUDA_VISIBLE_DEVICES=6 python mytrain_generative_pixelshift.py --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=unet --model_sig=wine
# CUDA_VISIBLE_DEVICES=7 python mytrain_generative_pixelshift.py --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=resnet --model_sig=wine

# python mytrain_generative_pixelshift.py --gpunum=5 --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=bwunet --model_sig=wine
# python mytrain_generative_pixelshift.py --gpunum=6 --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=bwunet --model_sig=wine
# python mytrain_generative_pixelshift.py --gpunum=7 --dataset_path=/home/team19/datasets --batch_size=8 --epoch=600 --model_name=bwunet --model_sig=wine


# from util.visualizer import Visualizer
from mymodel import mygen_model, mydisc_model, set_requires_grad
from myutils import LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

from mydataset import * #give_me_dataloader, SingleDataset, give_me_transform, give_me_test_images, give_me_comparison, degamma
from myloss import BayerLoss
import matplotlib.pyplot as plt

def save_model(modelG, ckpt_path, epoch, loss=0.0, state='valid'):
    try:
        fname = os.path.join(ckpt_path, "generation_epoch_%05d__loss_%05.3e.pth"%(epoch, loss))
        if os.path.exists(fname):
            fname = fname.split('.pth')[0] + f'_{state}_1.pth'
        torch.save(
                {
                    "model_generation_A2B": modelG.state_dict(),
                    "epoch"      : epoch,
                },
                fname,
        )
    except:
        print('something wrong......skip saving model at epoch ', epoch)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print('dev=', dev)

    print(args)
    # args
    model_name      = args.model_name
    model_sig       = args.model_sig
    model_type      = args.model_type
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
    print('model_sig = ', model_sig)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_name = ', dataset_name)
    print('dataset_path = ', dataset_path)

    # dataset
    base_path = os.path.join(dataset_path, dataset_name)
    print('base_path: ', base_path)

    # path
    mytype = 'C_3ch'
    train_path = os.path.join(base_path, 'train'+ mytype)
    valid_path = os.path.join(base_path, 'valid'+ mytype)
    test_path  = os.path.join(base_path, 'viz'  + mytype)
    viz_path   = os.path.join(base_path, 'viz'  + mytype)


    mydata_path = {'train': train_path,
                   'valid': valid_path,
                   'test' : test_path,
                   'viz'  : viz_path}
    for k,v in mydata_path.items():
        print(k,' : ', v)


    # transform
    BITS = 16
    transform = {'train': give_me_transform2('train', isnpy=True, bits=BITS),
                 'valid': give_me_transform2('valid', isnpy=True, bits=BITS),
                 'test' : give_me_transform2('test',  isnpy=True, bits=BITS),
                 'viz'  : give_me_transform2('viz' ,  isnpy=True, bits=BITS)}

    # dataloader
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train'], bits=BITS, mylen=-1), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid'], bits=BITS, mylen=-1), batch_size),
                  'test' : give_me_dataloader(SingleDataset(mydata_path['test'],  transform['test'] , bits=BITS, mylen=-1), batch_size),
                  'viz'  : give_me_dataloader(SingleDataset(mydata_path['viz'],   transform['viz']  , bits=BITS, mylen=-1), batch_size=10) }


    nsteps={}
    for state in ['train', 'valid', 'test', 'viz']:
        nsteps[state] = len(dataloader[state])
        print( '%s len(%s): ' % (state , nsteps[state] ))

    # model
    model_G_A2B = mygen_model(model_name).to(device) # RGB --> RAW


    ## ckpt save load if any
    ckpt_path_name      = f'checkpoint/{model_type}/{dataset_name}'
    ckpt_path_name      = os.path.join(ckpt_path_name, model_name+model_sig)
    ckpt_path_name_best = os.path.join(ckpt_path_name, model_name+model_sig+'_best')

    os.makedirs(ckpt_path_name,      exist_ok=True)
    os.makedirs(ckpt_path_name_best, exist_ok=True)

    ckpt_list = os.listdir(ckpt_path_name)
    print(ckpt_list)


    epoch = 0
    if (ckpt_path_name is not None) and \
        (len(ckpt_list) > 0) :
            ckpt_list.sort()
            ckpt_name = os.path.join(ckpt_path_name, ckpt_list[-1])
            print('ckpt name = ', ckpt_name)
            if os.path.isfile(ckpt_name) and 'pth' in ckpt_name[-4:]:
                checkpoint = torch.load(ckpt_name, map_location=device)
                model_G_A2B.load_state_dict(checkpoint["model_generation_A2B"])
                epoch = checkpoint["epoch"]
    else:
        save_model(model_G_A2B, ckpt_path_name, epoch, float('inf'))



    ## save onnx
    dummy_input_G = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(model_G_A2B.eval(), dummy_input_G,
                os.path.join('checkpoint', model_type, f"Generation_{model_name + model_sig}_{model_type}.onnx"))



    # visualize test images
    test_batch = next( iter(dataloader['test']))
    print('test_batch.shape', test_batch.shape)


    # train_batch = next( iter(dataloader['train']))
    # print('train_batch.shape', train_batch.shape)





    # exit()


    logpath = os.path.join('runs', model_name + '_' + model_type + model_sig)
    os.makedirs(logpath, exist_ok=True)
    summary = SummaryWriter(logpath)
    test_images = give_me_visualization(model_A2B=model_G_A2B,
                                        model_B2A=None,
                                        device='cpu', test_batch=test_batch, nomalize=True, beta_for_gamma=1/2.2)
    summary.add_image('Generated_pairs', test_images.permute(2,0,1), 0)
    plt.imshow(test_images)
    plt.title('Real RGB \t Real RAW \n Fake RGB \t Fake RAW')
    #plt.show()
    # exit()


    # make model in training mode
    model_G_A2B.train()

    # Loss
    criterion_bayer = BayerLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_generation = nn.MSELoss()
    criterion_GAN = nn.MSELoss() #  vanilla: nn.BCEWithLogitsLoss(), lsgan: nn.MseLoss()

    # Optimizer, Schedular
    optim_G = optim.Adam(model_G_A2B.parameters(), lr=args.lr, betas=(0.5, 0.999),
    )


    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G         = optim.lr_scheduler.LambdaLR(
                                    optimizer=optim_G,
                                    lr_lambda=lr_lambda)
    # Training
    # logger for tensorboard.

    disp_train = LossDisplayer(["G_train"])
    disp_valid = LossDisplayer(["G_valid"])
    disp = {'train':disp_train, 'valid':disp_valid}


    step = {'train':0, 'valid':0}

    # loss_best_G_valid = float('inf')
    loss_best_G = {'train':float('inf'), 'valid':float('inf')}
    loss_G_train_last = float('inf')

    beta_for_gamma = 1/2.2

    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")

        loss_G_total = {'train':0, 'valid':0}
        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, real_B in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))

                if state == 'train' and idx == 0:
                    # train mode
                    model_G_A2B.train()
                elif state == 'valid' and idx == 0:
                    # eval mode
                    model_G_A2B.eval()


                # degamma to make raw image
                real_B = real_B.to(device)
                real_A = gamma(real_B, device, beta=beta_for_gamma).to(device)


                # ----------------------------------
                # Forward
                # ----------------------------------
                fake_B = model_G_A2B(real_A)

                # ----------------------------------
                # Train Generators
                # ----------------------------------

                ## UnFreeze D when training Generators


                loss_generation = criterion_generation(fake_B, real_B)

                # combine loss and calculate gradients
                loss_G = 0
                # loss_G += args.lambda_generation *  loss_generation
                loss_G += loss_generation
                loss_G_train_last = loss_G # for save

                step[state] += 1
                if state == 'train':
                    optim_G.zero_grad()
                    loss_G.backward()
                    optim_G.step()
                else:
                    if idx == 0:
                        step['valid'] = step['train']

                ## accumulate generator loss in validataion to save best ckpt
                loss_G_total[state] += loss_G

                # -----------------
                # record loss for tensorboard
                # -----------------
                disp[state].record([loss_G])
                if step[state]%100==0 and idx>1:
                    avg_losses = disp[state].get_avg_losses()
                    summary.add_scalar(f"loss_G_{state}",           avg_losses[0], step[state])

                    print(f'{state} : epoch{epoch}, step{step[state]}------------------------------------------------------')
                    print('loss_G: %.3f, ' % avg_losses[0], end='')
                    disp[state].reset()


                    test_images = give_me_visualization(model_A2B=model_G_A2B,
                                        model_B2A=None,
                                        device=device, test_batch=test_batch, nomalize=True, beta_for_gamma=1/2.2)

                    summary.add_image('Generated_pairs', test_images.permute(2,0,1), step[state])

        else:

            loss_G_average = loss_G_total[state] / nsteps[state]
            if loss_best_G[state] > loss_G_average:
                print(f'best {state} ckpt updated!!!  old best {loss_best_G[state]} vs new best {loss_G_average}')
                loss_best_G[state] = loss_G_average
                summary.add_scalar(f"loss_best_G{state}", loss_best_G[state], step[state])
                save_model(model_G_A2B, ckpt_path_name_best, epoch, loss_best_G[state])




        # Step scheduler
        scheduler_G.step()


        # Save checkpoint for every 5 epoch
        if epoch % 5 == 0:
            save_model(model_G_A2B, ckpt_path_name, epoch, loss_G_train_last)


    print('done done')



def main(args):
    train(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='resnet', type=str,
                    choices=['resnet', 'unet', 'bwunet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--model_type', default="generative", type=str,
                    choices=['gan', 'cyclegan', 'generative'], help='(default=gan, training type, GAN, CycleGAN, Generative')
    argparser.add_argument('--dataset_name', default='pixelshift', type=str,
                    choices=['sidd', 'pixelshift', 'apple2orange'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path', default='/data/team19', type=str, help='(default=datasets')
    # argparser.add_argument('--dataset_path', default='datasets', type=str, help='(default=datasets')
    argparser.add_argument('--model_sig', default="_gogo",
                    type=str, help='(default=model signature for same momdel different ckpt/log path)')

    argparser.add_argument('--device', default='cuda', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=128)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=1)
    argparser.add_argument("--lambda_ide", type=float, default=10)
    argparser.add_argument("--lambda_generation", type=float, default=1)
    argparser.add_argument("--lambda_gan", type=float, default=100)
    argparser.add_argument("--lambda_cycle", type=float, default=5)
    argparser.add_argument("--identity", action="store_true")
    argparser.add_argument("--gpunum", default="0", type=str,)
    args = argparser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpunum
    args = argparser.parse_args()

    main(args)