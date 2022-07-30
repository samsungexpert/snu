import os
from glob import glob
import numpy as np
import torch, torchvision, torchsummary
from torch import nn

from torchvision import models, transforms


from models import networks








def mygen_model(model_name:str):
    print('network: ', model_name)

    valid_model_name = ('resnet', 'unet', 'bwunet')
    assert model_name.lower() in valid_model_name, 'model should be in ' + str(valid_model_name) + ' but \'%s\'' % model_name
    model = None


    if model_name == 'resnet':
        model = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks')

    elif model_name=='unet':
        model = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_128')

    elif model_name == 'bwunet':
        model = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='bwunet')


    return model


def mydisc_model(model_name:str='basic', input_nc=3):

    if model_name not in ['basic', 'n_layers', 'pixel']:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % model_name)

    model = networks.define_D(input_nc=input_nc, ndf=64, netD=model_name, n_layers_D=3)


    return model


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def main():
    # model = mygen_model('bwunet')
    # model = mygen_model('unet')
    # model =  mygen_model('resnet')

    model =  mydisc_model('basic').to('cuda')

    print(model)
    torchsummary.summary(model, input_size=(3, 128, 128))


if __name__ == '__main__':
    main()