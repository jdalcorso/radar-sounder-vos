import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import torchvision.models as models
import matplotlib.pyplot as plt

from dataset import SingleVideo
from model import RGVOS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import dot_product_attention

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS test', add_help=False)
    # Model parameters
    parser.add_argument('--image_size', default=(512,48), type=int)
    # Loss parameters
    
    # Training parameters
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1E-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--datafolder',default='/data/videos/class_0') #default='./cresis_of/train/sample'
    parser.add_argument('--savefolder',default='./radar_vos_run/') #default='./cresis_of/train/sample'
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RGVOS()
    model.to(device)
    model.load_state_dict(torch.load('./trained-vos-latest.pt'))

    # Imagenet transformation
    normalize = transforms.Normalize(mean = 0.45, std = 0.22)

    dataset = SingleVideo(args.datafolder)
    
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    loss_fn = nn.HuberLoss()

    # Initialize training
    print('Testing on:', device)
    if device.type == 'cuda':
        print('Total CUDA memory:',torch.cuda.get_device_properties(0).total_memory/1024**3)
        print('Reserved CUDA memory:',torch.cuda.memory_reserved(0)/1024**3)
        print('Allocated CUDA memory:',torch.cuda.memory_allocated(0)/1024**3)

    print('---------------------------------------------------------')
    print('START TESTING')
    model.eval()

    video, label = dataset[0]
    video = video.to(device)
    label = label.to(device)

    _,T,H,W = video.shape
    seg = torch.ones(H,T*W)
    seg[:,:W] = label

    for i in range(T-1):
        v = video[:,i:i+2,:,:].unsqueeze(0)

        # Imagenet normalization (if resnet18 is pretrained)
        sample1 = normalize(v[:,:,0,:,:])
        sample2 = normalize(v[:,:,1,:,:])
        v = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)

        x,y = model(v)
        fH = x.shape[2]
        fW = x.shape[3]
        # Label propagation
        x = x.squeeze(0)
        y = y.squeeze(0)

        A = dot_product_attention(x,y)

        downscale = transforms.Resize((fH,fW), interpolation=InterpolationMode.NEAREST)
        upscale = transforms.Resize((H,W), interpolation=InterpolationMode.NEAREST)

        label_dw = downscale(label.unsqueeze(0)).squeeze(0).view(-1)
        
        next_lbl = torch.matmul(A,label_dw.unsqueeze(1)).squeeze(1).view(fH,fW).unsqueeze(0)
        next_lbl = upscale(next_lbl)

        seg[:, W*(i+1):W*(i+1)+W] = next_lbl
        label = next_lbl

    print('--- TEST DONE ---')
    print('Saving results in lbl.png')
    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    plt.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)