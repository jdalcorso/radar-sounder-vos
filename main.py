import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import torchvision.models as models

from dataset import VideoDataset, VideoDataset2, SingleVideo
from model import RGVOS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from utils import dot_product_attention

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--image_size', default=(512,48), type=int)
    parser.add_argument('--patch_size', default=(8,8), type=int)
    parser.add_argument('--channels', default=1, type=int)
    # Loss parameters
    
    # Training parameters
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1E-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--datafolder',default='/data/videos/class_0') #default='/data/videos/class_0', '/data/videos24'
    parser.add_argument('--savefolder',default='./radar_vos_run/') #default='./cresis_of/train/sample'
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('./radar_vos/logs/')
    writer.add_text('arguments',str(args))

    model = RGVOS()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)

    dataset = VideoDataset2(args.datafolder)
    
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    # Initialize optimizer
    optimizer = AdamW(params=model.parameters() ,lr = args.lr, betas=(0.9, 0.95))
    loss_fn = nn.HuberLoss()

    # Imagenet transformation
    normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792]) # Computed on videos24
    # normalize = transforms.Compose([
    #     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    #     transforms.ColorJitter(brightness = 0.05, contrast = 0.05)
    # ])


    # Initialize training
    print('Training on:', device)
    if device.type == 'cuda':
        print('Total CUDA memory:',torch.cuda.get_device_properties(0).total_memory/1024**3)
        print('Reserved CUDA memory:',torch.cuda.memory_reserved(0)/1024**3)
        print('Allocated CUDA memory:',torch.cuda.memory_allocated(0)/1024**3)

    print('---------------------------------------------------------')
    print('START TRAINING FOR',args.epochs,'EPOCHS')
    model.train(True)

    train_loss = []
   
    for epoch in range(args.epochs):

        train_loss_epoch = []
        # Batch train loop
        for batch, (samples, _) in enumerate(dataloader):
            samples = samples.to(device)
            current_bs = samples.shape[0]

            # Imagenet normalization (if resnet18 is pretrained)
            sample1 = normalize(samples[:,:,0,:,:].repeat(1,3,1,1))
            sample2 = normalize(samples[:,:,1,:,:].repeat(1,3,1,1))
            samples = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)

            x,y = model(samples)

            downscale = transforms.Resize((x.shape[-2],x.shape[-1]), interpolation=InterpolationMode.NEAREST)
            upscale = transforms.Resize((sample1.shape[-2],sample1.shape[-1]), interpolation=InterpolationMode.NEAREST)

            downscaled_sample1 = downscale(sample1)
            downscaled_sample2 = downscale(sample2)

            # Pretext task, "colorization" of target image y starting from surce image x
            loss = 0
            for i in range(current_bs):
                A = dot_product_attention(x[i,...], y[i,...])
                reco = torch.matmul(A,downscaled_sample1[i,0,...].view(-1))
                reco = reco.view(*downscaled_sample1[i,0,...].shape)
                loss += loss_fn(downscaled_sample2[i,0,...],reco)
            loss = loss/current_bs

            # Loss between true target (sample2) and recolorized one
            # loss = LOSS(sample2, ...)
            #loss = loss_fn(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log loss of every first batch
            train_loss_epoch.append(loss.cpu().detach().numpy().item()) 

        train_loss.append(torch.tensor(train_loss_epoch).mean())
        writer.add_scalar('Train Loss', train_loss[-1], epoch)
        print('Epoch',epoch+1,'- Train loss:', train_loss[-1].item())
        
        writer.close()

    # Saving only the encoder
    torch.save(model.state_dict(), './trained-vos-test.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)