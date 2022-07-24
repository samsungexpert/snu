import os
from glob import glob
import numpy as np
import torch, torchvision, torchsummary
from torch import nn

from torchvision import models, transforms


from models import networks


def mymodel(model_name:str, input_size=(128,128)):
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



def main():
    # model = mymodel('bwunet')
    # model = mymodel('unet')
    model =  mymodel('resnet')

    print(model)
    torchsummary.summary(model, input_size=(3, 128, 128))


if __name__ == '__main__':
    main()