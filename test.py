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
from utils import dot_product_attention, combine_masks_to_segmentation

from imported.labelprop import LabelPropVOS_CRW

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS test', add_help=False)
    # Model parameters
    # Loss parameters
    # Training parameters
    parser.add_argument('--datafolder',default='/data/videos/class_0') #default='/data/videos/class_0'
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RGVOS()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./trained-vos-test.pt'))

    # Imagenet transformation
    normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792])

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
    seg = torch.zeros(H,T*W)
    seg[:,:W] = label

    cfg = {
        'CXT_SIZE' : 10, 
        'RADIUS' : 10,
        'TEMP' : 0.05,
        'KNN' : 5,
    }
    lp = LabelPropVOS_CRW(cfg)
    num_classes = 3

    feats = []
    masks = []

    for i in range(T-1):
        v = video[:,i:i+2,:,:].unsqueeze(0)

        # Imagenet normalization (if resnet18 is pretrained)
        sample1 = normalize(v[:,:,0,:,:].repeat(1,3,1,1))
        sample2 = normalize(v[:,:,1,:,:].repeat(1,3,1,1))
        v = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)

        x,y = model(v)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        fH = x.shape[2]
        fW = x.shape[3]

        downscale = transforms.Resize((fH,fW), interpolation=InterpolationMode.NEAREST)
        upscale = transforms.Resize((H,W), interpolation=InterpolationMode.NEAREST)

        # downscale label and turn into a mask
        label = downscale(label.unsqueeze(0)).squeeze(0)
        ctx = torch.zeros(num_classes, fH, fW, device = device)
        for class_idx in range(0, num_classes):
            mask = (label == class_idx).unsqueeze(0).float()
            ctx[class_idx, :, :] = mask
        ctx = ctx.unsqueeze(0)
        
        masks.append(ctx)
        feats.append(x)

        mask = lp.predict(feats = feats, masks = masks, curr_feat = y)
        next_lbl = combine_masks_to_segmentation(mask)
        next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)
        label = torch.round(next_lbl.squeeze(0))
        seg[:, W*(i+1):W*(i+1)+W] = label

    print('--- TEST DONE ---')
    print('Saving results in lbl.png')
    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    plt.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)