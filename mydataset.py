
import os
import numpy as np

import glob, random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms



# trandform
def give_me_transform(type):

    transform = None
    if type == 'train':
        transform = transforms.Compose(
            [
                # transforms.Resize((args.size, args.size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    return transform


def give_me_dataloader(dataset, batch_size:int, shuffle=True, num_workers=4, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


class SingleDataset(DataLoader):
    def __init__(self, dataset_dir, transforms):
        self.dataset_dir = dataset_dir
        self.image_path = glob.glob(os.path.join(dataset_dir, "**/*") , recursive=True)
        self.transform = transforms
        print('--------> # of images: ', len(self.image_path))

    def __getitem__(self, index):
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




def main():
    ...


if __name__ == '__main__':
    main()