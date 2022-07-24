import argparse

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model', default='resnet', type=str,
                    choices=['resnet', 'unet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset', default='sidd', type=str,
                    choices=['sidd','pixelshift'],
                    help='(default=%(default)s)')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch-size', type=int, help='mini batch size', default=128)
    args = argparser.parse_args()

    return args
