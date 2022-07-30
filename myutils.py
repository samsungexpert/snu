import random
import torch
from torch import nn
import matplotlib.pyplot as plt


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images




class LossDisplayer:
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.loss_list = [0] * len(self.name_list)

    def record(self, losses):
        self.count += 1
        for i, loss in enumerate(losses):
            self.loss_list[i] += loss.item()

    def get_avg_losses(self):
        return [loss / self.count for loss in self.loss_list]

    def display(self):
        for i, total_loss in enumerate(self.loss_list):
            avg_loss = total_loss / self.count
            print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.loss_list = [0] * len(self.name_list)


def init_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


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


if __name__ == '__main__':
    print()