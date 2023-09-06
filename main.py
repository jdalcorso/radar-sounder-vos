import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import torchvision.models as models

from dataset import VideoDataset, VideoDataset2
from model import RGVOS
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1E-6, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--datafolder',default='/data/videos/class_0') #default='./cresis_of/train/sample'
    parser.add_argument('--savefolder',default='./radar_vos_run/') #default='./cresis_of/train/sample'
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('./radar_vos/logs/')
    writer.add_text('arguments',str(args))

    model = RGVOS()
    model.to(device)

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
    normalize = transforms.Normalize(mean = 0.45, std = 0.22)

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
            
            # Imagenet normalization (if resnet18 is pretrained)
            sample1 = normalize(samples[:,:,0,:,:])
            sample2 = normalize(samples[:,:,1,:,:])
            samples = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)

            x,y = model(samples)
            loss = loss_fn(x,y)
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
    torch.save(model.state_dict(), './trained-vos-latest.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)