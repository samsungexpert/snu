
import torchvision
from torchvision import transforms, datasets


from torchvision.transforms.transforms import ConvertImageDtype
upsize = 286
data_transforms = {
    'train': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((upsize, upsize),interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    'test': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((upsize, upsize),interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ConvertImageDtype(torch.float),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}


def main():
    ...


if __name__ == '__main__':
    main()