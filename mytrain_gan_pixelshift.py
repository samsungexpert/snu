# https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo

import os, shutil
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim

# from util.visualizer import Visualizer
from mymodel import mygen_model, mydisc_model, set_requires_grad
from myutils import LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

from mydataset import * #give_me_dataloader, SingleDataset, give_me_transform, give_me_test_images, give_me_comparison, degamma
from myloss import BayerLoss
import matplotlib.pyplot as plt


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

    # try:
    #     shutil.rmtree('runs/gan')
    #     shutil.rmtree('checkpoint/gan')
    #     shutil.rmtree('runs')
    # except:
    #     pass

    # dataset
    base_path = os.path.join(dataset_path, dataset_name)
    print('base_path: ', base_path)

    # path
    mytype = 'A'
    mytype = 'C'
    mytype = ''
    train_path = os.path.join(base_path, 'train'+mytype)
    valid_path  = os.path.join(base_path, 'test'+mytype)
    test_path = os.path.join('imgs', dataset_name)


    mydata_path = {'train': train_path,
                   'valid': valid_path,
                   'test' : test_path}
    print(mydata_path)
    print('train_path: ', train_path)
    print('valid_path: ', valid_path)
    print('test_path: ', test_path)
    # transform
    transform = {'train': give_me_transform('train'),
                 'valid': give_me_transform('valid'),
                 'test' : give_me_transform('test')}

    # dataloader
    BITS = 14
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train'], bits=BITS), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid'], bits=BITS), batch_size),
                  'test' : give_me_dataloader(SingleDataset(mydata_path['test'],  transform['test'] , bits=BITS), batch_size=2) }


    nsteps={}
    for state in ['train', 'valid']:
        nsteps[state] = len(dataloader[state])
        print( '%s len(%s): ' % (state , len(dataloader[state]) ))

    # model
    model_G_rgb2raw = mygen_model(model_name).to(device)
    model_D_raw     = mydisc_model('basic', input_nc=6).to(device)


    ## ckpt save load if any
    ckpt_path_name = f'checkpoint/{model_type}/{dataset_name}'
    ckpt_path_name = os.path.join(ckpt_path_name, model_name+model_sig)
    # ckpt_path_name = f"checkpoint/{dataset_name}"
    os.makedirs(ckpt_path_name, exist_ok=True)

    ckpt_list = os.listdir(ckpt_path_name)
    if (ckpt_path_name is not None) and \
        (len(ckpt_list) > 0) :

        ckpt_list.sort()
        ckpt_name = os.path.join(ckpt_path_name, ckpt_list[-1])
        print('ckpt name = ', ckpt_name)
        checkpoint = torch.load(ckpt_name, map_location=device)

        model_G_rgb2raw.load_state_dict(checkpoint["model_G_rgb2raw"])
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
                        "model_D_raw": model_D_raw.state_dict(),
                        "epoch": epoch,
                    },
                    fname,
            )
        except:
            print('skip')



    dummy_input_G = torch.randn(1, 3, 256, 256, device=device)
    dummy_input_D = torch.randn(1, 6, 256, 256, device=device)

    torch.onnx.export(model_G_rgb2raw.eval(), dummy_input_G, f"G_{model_name + model_sig}_{model_type}.onnx")
    torch.onnx.export(model_D_raw.eval(), dummy_input_D,     f"D_{model_name + model_sig}_{model_type}.onnx")






    # visualize test images
    test_batch = next( iter(dataloader['test']))

    logpath = os.path.join('runs', model_name+model_sig)
    os.makedirs(logpath, exist_ok=True)
    summary = SummaryWriter(logpath)
    test_images = give_me_visualization(model_G_rgb2raw, None, 'cpu', test_batch, beta_for_gamma=1/2.2)
    summary.add_image('Generated_pairs', test_images.permute(2,0,1), 0)
    # plt.imshow(test_images)
    # plt.title('Real RGB \t Real RAW \n Fake RGB \t Fake RAW')
    # plt.show()



    # make model in training mode
    model_G_rgb2raw.train()
    model_D_raw.train()

    # Loss
    criterion_bayer = BayerLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_generation = nn.MSELoss()
    criterion_GAN = nn.MSELoss() #  vanilla: nn.BCEWithLogitsLoss(), lsgan: nn.MseLoss()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        model_G_rgb2raw.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optim_D_raw = optim.Adam(model_D_raw.parameters(), lr=args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G         = optim.lr_scheduler.LambdaLR(
                                    optimizer=optim_G,
                                    lr_lambda=lr_lambda)
    scheduler_D_rgb2raw = optim.lr_scheduler.LambdaLR(
                                    optimizer=optim_D_raw,
                                    lr_lambda=lr_lambda)


    # Training
    # logger for tensorboard.

    disp_train = LossDisplayer(["G_train",
                                "G_GAN_train",
                                "G_Identity_train",
                                "D_train"])
    disp_valid = LossDisplayer(["G_valid",
                                "G_GAN_valid",
                                "G_Identity_valid",
                                "D_valid"])
    disp = {'train':disp_train, 'valid':disp_valid}

    step = 0
    loss_best_G = 1e10

    beta_for_gamma=1/2.2

    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")


        # for state in ['train', 'valid']:
        for state in ['train']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, rgbimage in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))

                if state == 'train' and idx == 0:
                    # train mode
                    model_G_rgb2raw.train()
                    optim_G.zero_grad()
                elif state == 'valid' and idx == 0:
                    # eval mode
                    model_G_rgb2raw.eval()


                # degamma to make raw image
                real_rgb = rgbimage.to(device)
                real_raw = gamma(rgbimage, device, beta=beta_for_gamma).to(device)


                # -----------------
                # Train Generators
                # -----------------



                #### Forward
                fake_raw = model_G_rgb2raw(real_rgb)


                ## freeze D when training Generators
                set_requires_grad([model_D_raw], False)


                if state == 'train':
                    optim_G.zero_grad()


                loss_G = 0

                # Generation Loss
                generation_raw = model_G_rgb2raw(real_rgb)

                loss_generation_raw = criterion_generation(generation_raw, real_raw)
                loss_generation = loss_generation_raw

                loss_G += args.lambda_generation *  loss_generation


                # GAN loss
                din = torch.cat([fake_raw, real_rgb], axis=1)
                # print('din.shape', din.shape)
                pred_fake_raw = model_D_raw(din)

                loss_GAN_raw  = criterion_GAN(pred_fake_raw, torch.ones_like(pred_fake_raw))
                loss_GAN = loss_GAN_raw

                loss_G += args.lambda_gan * loss_GAN



                if state == 'train':
                    loss_G.backward()
                    optim_G.step()



                # -----------------
                # Discriminator Setup
                # -----------------
                if state == 'train':
                    optim_D_raw.zero_grad()
                    require_grad_for_D = True
                else:
                    require_grad_for_D = False

                 ## freeze D when training Generators
                set_requires_grad([model_D_raw, model_D_raw], require_grad_for_D)



                # -----------------
                # Train Discriminator Raw
                # -----------------
                din1 = torch.cat([real_raw, real_rgb], axis=1)
                din0 = torch.cat([real_raw, real_rgb], axis=1)
                pred_real_raw = model_D_raw(din1)
                pred_fake_raw = model_D_raw(din0.detach())

                loss_D_raw_real = criterion_GAN(pred_real_raw, torch.ones_like( pred_real_raw))
                loss_D_raw_fake = criterion_GAN(pred_fake_raw, torch.zeros_like(pred_fake_raw))
                loss_D_raw = 0.5 * ( loss_D_raw_real + loss_D_raw_fake )


                # ----------------

                loss_D = loss_D_raw



                # backward for discriminator
                if state == 'train':
                    loss_D_raw.backward()
                    optim_D_raw.step()
                    step+=1
                    if loss_best_G > loss_G:
                        loss_best_G = loss_G
                        summary.add_scalar(f"loss_best_G{state}", loss_best_G, step)



                # -----------------
                # record loss for tensorboard
                # -----------------
                disp[state].record([loss_G, loss_GAN, loss_generation, loss_D])
                if step%100==0 :
                    avg_losses = disp[state].get_avg_losses()
                    summary.add_scalar(f"loss_G_{state}",           avg_losses[0], step)
                    summary.add_scalar(f"loss_G_G_GAN_{state}",     avg_losses[1], step)
                    summary.add_scalar(f"loss_G_Generation_{state}",  avg_losses[2], step)
                    summary.add_scalar(f"loss_D{state}",     avg_losses[3], step)

                    print(f'{state} : epoch{epoch}, step{step}------------------------------------------------------')
                    print('loss_G: %.3f, ' % avg_losses[0], end='')
                    print('loss_G_G_GAN: %.3f, ' % avg_losses[1], end='')
                    print('loss_G_Generation: %.3f, ' % avg_losses[2], end='')
                    print('loss_D: %.3f' % avg_losses[3])

                    disp[state].reset()

                    # log images
                    test_images = give_me_visualization(model_G_rgb2raw, None, device, test_batch)
                    summary.add_image('Generated_pairs', test_images.permute(2,0,1), step)


        # Step scheduler
        scheduler_G.step()
        scheduler_D_rgb2raw.step()


        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "model_G_rgb2raw": model_G_rgb2raw.state_dict(),
                    "model_D_raw":     model_D_raw.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(ckpt_path_name, "gan_%05d.pth"%epoch),
            )


    print('done done')



def main(args):
    train(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='resnet', type=str,
                    choices=['resnet', 'unet', 'bwunet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--model_type', default="gan", type=str,
                    choices=['gan', 'cyclegan', 'generative'], help='(default=gan, training type, GAN, CycleGAN, Generative')
    argparser.add_argument('--dataset_name', default='pixelshift', type=str,
                    choices=['sidd', 'pixelshift', 'apple2orange'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path', default=os.path.join('datasets'), type=str,
                    help='(default=datasets')
    argparser.add_argument('--model_sig', default="_hello",
                    type=str, help='(default=model signature for same momdel different ckpt/log path)')

    argparser.add_argument('--device', default='cuda', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=128)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)
    argparser.add_argument("--lambda_ide", type=float, default=10)
    argparser.add_argument("--lambda_generation", type=float, default=1)
    argparser.add_argument("--lambda_gan", type=float, default=100)
    argparser.add_argument("--lambda_cycle", type=float, default=5)
    argparser.add_argument("--identity", action="store_true")
    args = argparser.parse_args()
    main(args)