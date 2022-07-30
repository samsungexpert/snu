
import numpy as np

import torch
from torch import nn

crop_size  = 256
cfa_pattern = 1
idx_R = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G1 = np.tile(
        np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G2 = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_B = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G = np.tile(
        np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))


idx_RGB = np.concatenate((idx_R[np.newaxis, ...],
                          idx_G[np.newaxis, ...],
                          idx_B[np.newaxis, ...]), axis=0)

idx_G1RBG2 = np.concatenate((idx_G1[np.newaxis, ...],
                             idx_R [np.newaxis, ...],
                             idx_B [np.newaxis, ...],
                             idx_G2[np.newaxis, ...]), axis=0)


idx_R      = torch.unsqueeze(torch.Tensor(idx_R), dim=0)
idx_G1     = torch.unsqueeze(torch.Tensor(idx_G1), dim=0)
idx_G2     = torch.unsqueeze(torch.Tensor(idx_G2), dim=0)
idx_G      = torch.unsqueeze(torch.Tensor(idx_G), dim=0)
idx_B      = torch.unsqueeze(torch.Tensor(idx_B), dim=0)
idx_RGB    = torch.unsqueeze(torch.Tensor(idx_RGB), dim=0)
idx_G1RBG2 = torch.unsqueeze(torch.Tensor(idx_G1RBG2), dim=0)

class BayerLoss(nn.Module):
    def __init__(self, norm='l2', crop_size=128, cfa_pattern=1):
        super(BayerLoss, self).__init__()
        self.crop_size = crop_size
        self.cfa_pattern = cfa_pattern

        norms = ['l1', 'l2']
        assert norm in norms, 'norm should be ' + norms

        if norm == 'l1':
            self.norm = nn.L1Loss()
        elif norm =='l2':
            self.norm = nn.MSELoss()
        else:
            ValueError('norm not in ', norms)

    def get_patternized_1ch_raw_image(self, image):
        patternized = self.get_patternized_3ch_image(image)
        patternized = torch.unsqueeze(torch.sum(patternized, dim=1), dim=0)
        return patternized

    def get_patternized_3ch_image(self, image):
        RGB = idx_RGB.type(torch.float32)
        print(type(image), image.shape)
        print(type(RGB), RGB.shape)
        patternized = torch.mul(image, RGB)
        return patternized

    def forward(self, inputs, targets):
        inputs_raw  = self.get_patternized_1ch_raw_image(inputs)
        targets_raw = self.get_patternized_1ch_raw_image(targets)
        return self.norm(targets_raw, inputs_raw)







if __name__ == '__main__':
    pass
    # # print(idx_R.shape)
    # # print(idx_RGB.shape)


    # input = np.arange(48).reshape((1,3,4,4))
    # input = torch.Tensor(input)
    # # print(input)

    # bayerLoss = BayerLoss('l2')


    # # i3 = bl.get_patternized_3ch_image(input)
    # # print(i3, i3.shape)

    # # i1 = bl.get_patternized_1ch_raw_image(input)
    # # print(i1, i1.shape)

    # img1 = np.arange(48).reshape((1,3,4,4))
    # img1 = torch.Tensor(img1)
    # img1 = bayerLoss.get_patternized_1ch_raw_image(img1)

    # img2 = np.arange(48).reshape((1,3,4,4))-47
    # img2 = torch.Tensor(img2)
    # img2 = bayerLoss.get_patternized_1ch_raw_image(img2)

    # print(img1, img1.shape)
    # print(img2, img2.shape)

    # l = bayerLoss(img1, img2)
    # print(l)

    # print(abs(torch.mean((img1-img2)**2)))



