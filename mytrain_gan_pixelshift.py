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

def save_model(modelG, modelD, ckpt_path, epoch):
    ...
    try:
        fname = os.path.join(ckpt_path, "{%05d}.pth"%epoch)
        if os.path.exists(fname):
            fname = fname.split('.pth')[0] + '_1.pth'
        torch.save(
                {
                    "model_G_A2B": modelG.state_dict(),
                    "model_D_B"  : modelD.state_dict(),
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

    # try:``
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
    train_path = os.path.join(base_path, 'train'+mytype)
    valid_path = os.path.join(base_path, 'valid'+mytype)
    test_path  = os.path.join(base_path, 'test' +mytype)
    viz_path   = os.path.join(base_path, 'viz' +mytype)


    mydata_path = {'train': train_path,
                   'valid': test_path, #valid_path,
                   'test' : viz_path,  #test_path,
                   'viz'  : viz_path}
    for k,v in mydata_path.items():
        print(k,' : ', v)


    # transform
    transform = {'train': give_me_transform('train'),
                 'valid': give_me_transform('valid'),
                 'test' : give_me_transform('test'),
                 'viz' : give_me_transform('viz')}

    # dataloader
    BITS = 14
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train'], bits=BITS), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid'], bits=BITS), batch_size),
                  'test' : give_me_dataloader(SingleDataset(mydata_path['test'],  transform['test'] , bits=BITS), batch_size),
                  'viz'  : give_me_dataloader(SingleDataset(mydata_path['test'],  transform['viz']  , bits=BITS), batch_size=2) }


    nsteps={}
    for state in ['train', 'valid']:
        nsteps[state] = len(dataloader[state])
        print( '%s len(%s): ' % (state , len(dataloader[state]) ))

    # model
    model_G_A2B = mygen_model(model_name).to(device) # RGB --> RAW
    model_D_B   = mydisc_model('basic', input_nc=6).to(device)


    ## ckpt save load if any
    ckpt_path_name = f'checkpoint/{model_type}/{dataset_name}'
    ckpt_path_name_best = os.path.join(ckpt_path_name, model_name+model_sig+'_best')
    ckpt_path_name = os.path.join(ckpt_path_name, model_name+model_sig)

    os.makedirs(ckpt_path_name, exist_ok=True)
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
                model_G_A2B.load_state_dict(checkpoint["model_G_A2B"])
                model_D_B.load_state_dict(checkpoint["model_D_B"])
                epoch = checkpoint["epoch"]
    else:
        save_model(model_G_A2B, model_D_B, ckpt_path_name, epoch)



    ## save onnx
    dummy_input_G = torch.randn(1, 3, 256, 256, device=device)
    dummy_input_D = torch.randn(1, 6, 256, 256, device=device)

    torch.onnx.export(model_G_A2B.eval(), dummy_input_G, f"G_{model_name + model_sig}_{model_type}.onnx")
    torch.onnx.export(model_D_B.eval(),   dummy_input_D, f"D_{model_name + model_sig}_{model_type}.onnx")


    # visualize test images
    test_batch = next( iter(dataloader['test']))

    logpath = os.path.join('runs', model_name+model_sig)
    os.makedirs(logpath, exist_ok=True)
    summary = SummaryWriter(logpath)
    test_images = give_me_visualization(model_A2B=model_G_A2B,
                                        model_B2A=None,
                                        device='cpu', test_batch=test_batch, nomalize=True, beta_for_gamma=1/2.2)
    summary.add_image('Generated_pairs', test_images.permute(2,0,1), 0)
    # plt.imshow(test_images)
    # plt.title('Real RGB \t Real RAW \n Fake RGB \t Fake RAW')
    # plt.show()



    # make model in training mode
    model_G_A2B.train()
    model_D_B.train()

    # Loss
    criterion_bayer = BayerLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_generation = nn.MSELoss()
    criterion_GAN = nn.MSELoss() #  vanilla: nn.BCEWithLogitsLoss(), lsgan: nn.MseLoss()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        model_G_A2B.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optim_D_B = optim.Adam(model_D_B.parameters(), lr=args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G         = optim.lr_scheduler.LambdaLR(
                                    optimizer=optim_G,
                                    lr_lambda=lr_lambda)
    scheduler_D_B = optim.lr_scheduler.LambdaLR(
                                    optimizer=optim_D_B,
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


    step = {'train':0, 'valid':0}
    loss_best_G_valid = float('inf')

    beta_for_gamma = 1/2.2

    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")

        loss_G_total = 0
        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, real_B in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))

                if state == 'train' and idx == 0:
                    # train mode
                    model_G_A2B.train()
                    model_D_B.train()
                elif state == 'valid' and idx == 0:
                    # eval mode
                    model_G_A2B.eval()
                    model_D_B.eval()


                # degamma to make raw image
                real_B = real_B.to(device)
                real_A = gamma(real_B, device, beta=beta_for_gamma).to(device)


                # ----------------------------------
                # Forward
                # ----------------------------------
                fake_B = model_G_A2B(real_A) # ok



                # ----------------------------------
                # Train Discriminator Raw
                # ----------------------------------

                ## freeze D when training Generators
                require_grad_for_D = True if state=='train' else False
                set_requires_grad([model_D_B], require_grad_for_D)
                if state == 'train':
                    optim_D_B.zero_grad()


                ## backward D
                # fake
                fake_AB   = torch.cat((real_A, fake_B), axis=1)
                pred_fake = model_D_B(fake_AB.detach())
                loss_D_fake = criterion_GAN(pred_fake, torch.ones_like( pred_fake))
                # real
                real_AB   = torch.cat((real_A, real_B), axis=1)
                pred_real = model_D_B(real_AB)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like( pred_fake))

                loss_D_raw = (loss_D_fake + loss_D_real) * 0.5

                # backward for discriminator
                if state == 'train':
                    loss_D_raw.backward()
                    optim_D_B.step()


                # ----------------------------------
                # Train Generators
                # ----------------------------------

                ## UnFreeze D when training Generators
                require_grad_for_D = False
                set_requires_grad([model_D_B], require_grad_for_D)
                if state == 'train':
                    optim_G.zero_grad()

                # First, G(A) should fake the discriminator
                fake_AB = torch.cat((real_A, fake_B), axis=1)
                pred_fake = model_D_B(fake_AB)
                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like( pred_fake))

                # Second, G(A) = B
                loss_generation = criterion_identity(fake_B, real_B)

                # combine loss and calculate gradients
                loss_G = 0
                loss_G += args.lambda_generation *  loss_generation
                loss_G += args.lambda_gan * loss_G_GAN

                step[state] += 1
                if state == 'train':
                    loss_G.backward()
                    optim_G.step()
                else:
                    if idx == 0:
                        step['valid'] = step['train']





                ## accumulate generator loss in validataion to save best ckpt
                if state == 'valid':
                    loss_G_total += loss_G

                # -----------------
                # record loss for tensorboard
                # -----------------
                disp[state].record([loss_G, loss_G_GAN, loss_generation, loss_D_raw])
                if step[state]%100==0 and idx>1:
                    avg_losses = disp[state].get_avg_losses()
                    summary.add_scalar(f"loss_G_{state}",           avg_losses[0], step[state])
                    summary.add_scalar(f"loss_G_G_GAN_{state}",     avg_losses[1], step[state])
                    summary.add_scalar(f"loss_G_Generation_{state}",avg_losses[2], step[state])
                    summary.add_scalar(f"loss_D{state}",            avg_losses[3], step[state])

                    print(f'{state} : epoch{epoch}, step{step[state]}------------------------------------------------------')
                    print('loss_G: %.3f, ' % avg_losses[0], end='')
                    print('loss_G_G_GAN: %.3f, ' % avg_losses[1], end='')
                    print('loss_G_Generation: %.3f, ' % avg_losses[2], end='')
                    print('loss_D: %.3f' % avg_losses[3])

                    disp[state].reset()


                    test_images = give_me_visualization(model_A2B=model_G_A2B,
                                        model_B2A=None,
                                        device=device, test_batch=test_batch, nomalize=True, beta_for_gamma=1/2.2)

                    summary.add_image('Generated_pairs', test_images.permute(2,0,1), step[state])

        else:
            if state=='valid':
                loss_G_average = loss_G_total / len(dataloader[state])
                if loss_best_G_valid > loss_G_average:
                    print(f'best ckpt updated!!!  old best {loss_best_G_valid} vs new best {loss_G_average}')
                    loss_best_G_valid = loss_G_average
                    summary.add_scalar(f"loss_best_G{state}", loss_best_G_valid, step[state])
                    save_model(model_G_A2B, model_D_B, ckpt_path_name_best, epoch)



        # Step scheduler
        scheduler_G.step()
        scheduler_D_B.step()


        # Save checkpoint for every 5 epoch
        if epoch % 5 == 0:
            save_model(model_G_A2B, model_D_B, ckpt_path_name, epoch)


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
    argparser.add_argument('--model_sig', default="_damn",
                    type=str, help='(default=model signature for same momdel different ckpt/log path)')

    argparser.add_argument('--device', default='cuda', type=str,
                    choices=['cpu','cuda'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=128)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=35)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)
    argparser.add_argument("--lambda_ide", type=float, default=10)
    argparser.add_argument("--lambda_generation", type=float, default=1)
    argparser.add_argument("--lambda_gan", type=float, default=100)
    argparser.add_argument("--lambda_cycle", type=float, default=5)
    argparser.add_argument("--identity", action="store_true")
    args = argparser.parse_args()
    main(args)