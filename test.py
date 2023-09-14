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
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS test', add_help=False)
    parser.add_argument('--image_size', default=(512,48), type=int)
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
    model.load_state_dict(torch.load('./trained-vos-latest.pt'))

    # Imagenet transformation
    normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792])

    resize2resnet = transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST)
    resize2frame = transforms.Resize(args.image_size, antialias = True, interpolation=InterpolationMode.NEAREST)

    dataset = SingleVideo(args.datafolder)
    
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    # Initialize training
    print('Testing on:', device)
    if device.type == 'cuda':
        print('Total CUDA memory:',torch.cuda.get_device_properties(0).total_memory/1024**3)
        print('Reserved CUDA memory:',torch.cuda.memory_reserved(0)/1024**3)
        print('Allocated CUDA memory:',torch.cuda.memory_allocated(0)/1024**3)

    print('---------------------------------------------------------')
    print('START TESTING')
    model.eval()

    kmeans = KMeans(3, n_init='auto', random_state=1)
    #kmeans = DBSCAN(eps=3,min_samples=30)
    #kmeans = SpectralClustering(3)

    video, label = dataset[0]
    video = video.to(device)
    label = label.to(device)

    _,T,H,W = video.shape
    seg = torch.zeros(H,T*W)
    seg[:,:W] = label

    segk = torch.zeros(H,T*W)
    segk[:,:W] = label

    cfg = {
        'CXT_SIZE' : 10, 
        'RADIUS' : 5,
        'TEMP' : 0.05,
        'KNN' : 10,
    }
    lp = LabelPropVOS_CRW(cfg)
    num_classes = 3

    feats = []
    masks = []

    for i in range(T-1):
        print('Range-line:',i)
        v = video[:,i:i+2,:,:].unsqueeze(0).detach()

        # Imagenet normalization (if resnet18 is pretrained)
        sample1 = normalize(v[:,:,0,:,:].repeat(1,3,1,1))
        sample2 = normalize(v[:,:,1,:,:].repeat(1,3,1,1))

        sample1 = resize2resnet(sample1)
        sample2 = resize2resnet(sample2)

        v = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
        
        with torch.inference_mode():
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

        kmeans_feats = torch.permute(y.squeeze(0).view(512,-1).cpu().detach(),[1,0])
        kmeans_res = torch.tensor(kmeans.fit(kmeans_feats).labels_).view(56,56)
        kmeans_res = upscale(kmeans_res.unsqueeze(0)).squeeze(0)
        segk[:, W*(i+1):W*(i+1)+W] = kmeans_res

    print('--- TEST DONE ---')
    print('Saving results in lbl.png')
    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    plt.close()

    plt.imshow(segk.cpu().detach())
    plt.savefig('lblk.png')
    plt.close()

    one_video_feats = x.squeeze(0)
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    plt.figure(figsize=(26,26))
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(one_video_feats[ix-1, :, :].cpu().detach(), cmap='gray')
            ix += 1
    # show the figure
    plt.show()    
    plt.savefig('lfeats.png')
    plt.close()

    # Saving only the encoder
    torch.save(model.state_dict(), './trained-vos-resize.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)