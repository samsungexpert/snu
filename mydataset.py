
import os
import numpy as np
import torch
import glob, random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def give_me_test_images(input_size=256):
    fname = ['apple.jpg', 'orange.jpg']
    image_tensor = torch.zeros(len(fname), 3, input_size, input_size)
    # image_list = []
    for idx, f in enumerate(fname):
        fpath = os.path.join('imgs', f)
        image = torch.Tensor(np.array(Image.open(fpath))).detach()
        image_tensor[idx] = image.permute(2,0,1)
    return image_tensor


def give_me_comparison(model, inputs, device):
    # print('inputs.size ', inputs.size(), type(inputs))
    with torch.no_grad():
        model.eval()
        if device == torch.device('cuda'):
            inputs=inputs.cuda()
            # print('input is in cuda')
        else:
            # print('input is in cpu')
            ...

        # print(type(inputs))
        # model.cpu()
        if device=='cuda' or next(model.parameters()).is_cuda:
            inputs=inputs.cuda()
        outputs = model(inputs)
    return outputs

def degamma(image, device, alpha=0.05):
    ## image ~ [-1, 1]
    ## RGB2RAW
    # gamma = torch.tensor(np.array([2.2]))
    # offsets = torch.tensor([1-torch.abs(torch.randn(1)), 1, torch.abs(torch.randn(1))])

    # print(image.shape, gamma.shape, offsets.shape)
    # gamma = torch.ones_like(image) * gamma
    # gammas = offsets * gamma
    beta = 2.2
    gammas = 1 + np.random.randn(3,1)*alpha
    gammas *= beta
    gammas[1] = beta
    # gammas = gammas * beta
    gammas1 = np.copy(gammas)
    gammas = torch.from_numpy(gammas)
    gammassss = torch.from_numpy(gammas1)
    # print('gammas.shape', gammas.shape)
    gammas = gammas.unsqueeze(0)
    gammas = gammas.unsqueeze(-1)
    if device=='cuda':
        gammas=gammas.cuda()

    # print('image.shape', image.shape)
    # print('gammas.shape', gammas.shape, '\n', gammas)
    image = (((image+1)/2)**(gammas))*2 - 1

    return image.float()



def gamma(image, device, beta=(1/2.2), alpha=0.05):
    ## image ~ [-1, 1] torch.tensor
    ## RAW2RGB

    gammas = 1 + np.random.randn(3,1)*alpha
    gammas *= beta
    gammas[1] = beta
    gammas = torch.from_numpy(gammas)
    gammas = gammas.unsqueeze(0)
    gammas = gammas.unsqueeze(-1)
    if device=='cuda':
        gammas=gammas.cuda()

    # print('image.shape', image.shape)
    # print('gammas.shape', gammas.shape, '\n', gammas)
    image = (((image+1)/2)**(gammas)) * 2 - 1

    return image.float()


def give_me_visualization(model_rgb2raw, model_raw2rgb=None, device='cpu', test_batch=None, nomalize=True, beta_for_gamma=2.2):
    # visualize test images
    # print('test_batch', type(test_batch))
    if test_batch != None:
        real_rgb_images = test_batch.cpu()
    else:
        real_rgb_images = give_me_test_images().to(device)
    real_raw_images = gamma(real_rgb_images, device, beta_for_gamma)
    if model_raw2rgb == None:
        fake_rgb_images = torch.zeros_like(real_raw_images)
    else:
        fake_rgb_images = give_me_comparison(model_raw2rgb, real_raw_images.to(device), device=device)
    fake_raw_images = give_me_comparison(model_rgb2raw, real_rgb_images.to(device), device=device)
    print('real_rgb (%.3f, %.3f), ' %(torch.amin(real_rgb_images), torch.amax(real_rgb_images)), end='')
    print('real_raw (%.3f, %.3f), ' %(torch.amin(real_raw_images), torch.amax(real_raw_images)), end='')
    print('fake_rgb (%.3f, %.3f), ' %(torch.amin(fake_rgb_images), torch.amax(fake_rgb_images)), end='')
    print('fake_raw (%.3f, %.3f), ' %(torch.amin(fake_raw_images), torch.amax(fake_raw_images)))

    real_rgb_images = vutils.make_grid(real_rgb_images, padding=2, normalize=nomalize)
    real_raw_images = vutils.make_grid(real_raw_images, padding=2, normalize=nomalize)
    fake_rgb_images = vutils.make_grid(fake_rgb_images, padding=2, normalize=nomalize)
    fake_raw_images = vutils.make_grid(fake_raw_images, padding=2, normalize=nomalize)

    real_images = torch.cat((real_rgb_images,real_raw_images ), dim=2)
    fake_images = torch.cat((fake_rgb_images.cpu(),fake_raw_images.cpu() ), dim=2)
    test_images = torch.cat((real_images.cpu(), fake_images.cpu()), dim=1)

    # if test_batch != None:
    test_images = test_images.permute(1,2,0)
    return test_images




# trandform
def give_me_transform(type, mean=0.5, std=0.5):

    transform = None
    if type == 'train':
        transform = transforms.Compose(
            [
                # transforms.Resize((args.size, args.size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    return transform


def give_me_dataloader(dataset, batch_size:int, shuffle=True, num_workers=4, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


class SingleDataset(DataLoader):
    def __init__(self, dataset_dir, transforms, mylen=-1):
        self.dataset_dir = dataset_dir
        self.transform = transforms
        self.mylen = mylen

        self.image_path = glob.glob(os.path.join(dataset_dir, "**/*") , recursive=True)
        if mylen>0:
            self.image_path = self.image_path[:mylen]

        print('--------> # of images: ', len(self.image_path))

    def __getitem__(self, index):
        if self.image_path[index].split('.')[-1] == 'npy':
            item = np.load(self.image_path[index]).astype(np.float32)
            # print('item.shape', item.shape)
            # exit()
            item = self.transform(item[...,(0,1,3)])
        else:
            item = self.transform(Image.open(self.image_path[index]))
        return item

    def __len__(self):
        return len(self.image_path)



class UnpairedDataset(DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        self.dataset_dir = dataset_dir
        self.styles = styles
        self.image_path_A = glob.glob(os.path.join(dataset_dir, styles[0]) + "/*")
        self.image_path_B = glob.glob(os.path.join(dataset_dir, styles[1]) + "/*")
        self.transform = transforms

    def __getitem__(self, index_A):
        index_B = random.randint(0, len(self.image_path_B) - 1)

        item_A = self.transform(Image.open(self.image_path_A[index_A]))
        item_B = self.transform(Image.open(self.image_path_B[index_B]))

        return [item_A, item_B]

    def __len__(self):
        return len(self.image_path_A)


def gamma_example():
    test_images= give_me_test_images()
    print(type(test_images[0]))
    test_images = ((test_images/255.)-0.5)*2
    print('test_images.shape', test_images[0].shape, type(test_images[0]), torch.min(test_images[0]), torch.max(test_images[0]))
    fake_rgb=[]
    fake_raw=[]

    device = 'cpu'
    for rgb in test_images:
        ...
        rgb = torch.tensor(rgb).to(device)
        raw =  degamma(rgb.unsqueeze(0), 'cpu').to(device).squeeze()
        rgb_ressc = gamma(raw, 'cpu').to(device).squeeze()
        print('rgb.shape', rgb.shape, type(rgb), torch.min(rgb), torch.max(rgb))
        print('raw.shape', raw.shape, type(raw), torch.min(raw), torch.max(raw))
        print('rgb_ressc.shape', rgb_ressc.shape, type(rgb_ressc), torch.min(rgb_ressc), torch.max(rgb_ressc))

        f = lambda x: (x+1)/2


        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(f(rgb.permute(1,2,0)))
        plt.title('rgb')

        plt.subplot(1,3,2)
        plt.imshow(raw.permute(1,2,0))
        plt.title('raw - degamma')

        plt.subplot(1,3,3)
        plt.imshow(rgb_ressc.permute(1,2,0))
        plt.title('rgb_resc -  gamma')

        plt.show()
        exit()

def degamma_example():
    test_images= give_me_test_images()
    print(type(test_images[0]))
    fake_rgb=[]
    fake_raw=[]

    device = 'cpu'
    for rgb in test_images:
        ...
        rgb = torch.tensor(rgb).to(device)
        raw = degamma(rgb.unsqueeze(0)).to(device).squeeze()
        print('rgb.shape', rgb.shape, type(rgb))
        print('raw.shape', raw.shape, type(raw))
        # raw_fake = model_G_rgb2raw(rgb)
        # rgb_fake = model_G_raw2rgb(raw)



    # degamma test

    image = torch.ones((2, 3, 2, 2))*.9
    print(image.shape)
    image_degamma = degamma(image)
    print(image_degamma.shape)
    print(image_degamma)


def main():
    # degamma_example()
    gamma_example()




if __name__ == '__main__':
    main()