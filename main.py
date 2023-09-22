import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import torchvision.models as models
import time

from dataset import VideoDataset, VideoDataset2, SingleVideo, MCORDS1Dataset, SingleVideoMCORDS1
from model import RGVOS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from utils import dot_product_attention, SupConLoss, runid, positional_encoding, plot_feats, plot_pca, SobelSmoothingLoss

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--image_size', default=(400,48), type=int) # Change this, if you change args.which_data
    parser.add_argument('--which_data', default = 1, type=int, help = '0 for MCORDS1_2010, 1 for Miguel ds')
    # Loss parameters
    parser.add_argument('--supconloss_w', default=0.1, type=float)
    parser.add_argument('--smoothloss_w', default=0.0, type=float)
    # Training parameters
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1E-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    # Plots and folders
    parser.add_argument('--pos_encode', default = False, type = bool)
    parser.add_argument('--plot_feats', default = True, type = bool)
    parser.add_argument('--plot_pca', default = True, type = bool)
    parser.add_argument('--datafolder',default='/data/videos/class_0') #default='/data/videos/class_0', '/data/videos24'
    parser.add_argument('--savefolder',default='./radar_vos_run/') #default='./cresis_of/train/sample'
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_id = runid()
    writer = SummaryWriter('./radar_vos/logs/'+run_id)
    writer.add_text('arguments',str(args))

    model = RGVOS()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)

    # Standard transformations between resnet18 size and image size
    resize2resnet = transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST)
    resize2frame = transforms.Resize(args.image_size, antialias = True, interpolation=InterpolationMode.NEAREST) 

    # Choose dataset, Imagenet transformation and single reference video according to arguments
    if args.which_data == 0:
        dataset = MCORDS1Dataset(factor = 1)
        normalize = transforms.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0])
        one_video = SingleVideoMCORDS1()
        one_video, one_map = one_video[0]
        one_video = one_video[0,0,:,:].unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
        one_video = normalize(one_video)
        one_video = resize2resnet(one_video)
    else: 
        dataset = VideoDataset2('/data/videos/class_0') # another option is '/data/videos24'
        normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792]) # Computed on videos24
        #normalize = transforms.Normalize(mean = [-534.5786, -534.5786, -534.5786], std = [154.9227, 154.9227, 154.9227])
        one_video = SingleVideo()
        one_video, one_map = one_video[0]
        one_video = one_video[0,0,:,:].unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
        one_video = normalize(one_video)
        one_video = resize2resnet(one_video)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    # Initialize optimizer
    optimizer = AdamW(params=model.parameters() ,lr = args.lr, betas=(0.9, 0.95))
    loss_fn = nn.HuberLoss()
    loss_fn2 = SupConLoss()
    loss_fn3 = SobelSmoothingLoss()

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
        t = time.time()
        train_loss_epoch = []
        # Batch train loop
        for batch, (samples, _) in enumerate(dataloader):
            samples = samples.to(device)
            current_bs = samples.shape[0]
            # Imagenet normalization (if resnet18 is pretrained)
            sample1 = normalize(samples[:,:,0,:,:].repeat(1,3,1,1))
            sample2 = normalize(samples[:,:,1,:,:].repeat(1,3,1,1))

            sample1 = resize2resnet(sample1)
            sample2 = resize2resnet(sample2)

            plt.imshow(sample1.cpu().detach()[0,0,...])
            plt.savefig('sample.png')
            plt.close()

            samples = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
            x,y = model(samples)
            bsz, fts, h, w = x.shape

            # --- POSITIONAL ENCODING ---
            if args.pos_encode:
                x = torch.permute(positional_encoding(torch.permute(x.view(bsz,fts,-1),[0,2,1]), fts),[0,2,1]).view(*x.shape)
                y = torch.permute(positional_encoding(torch.permute(y.view(bsz,fts,-1),[0,2,1]), fts),[0,2,1]).view(*y.shape)

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

            # ---- ONE-MAP LOSS ----
            supconloss = torch.tensor(0)
            if args.supconloss_w > 0:
                one_video_feats = model(one_video)
                if args.pos_encode:
                    one_video_feats = torch.permute(positional_encoding(torch.permute(one_video_feats.view(1,fts,-1),[0,2,1]), fts),[0,2,1]).view(*one_video_feats.shape)
                one_video_feats = one_video_feats.squeeze(0).view(fts,-1)
                one_map_ds = downscale(one_map.unsqueeze(0)).squeeze(0).view(-1)
                groups = torch.unique(one_map_ds)
                grouped_features = [one_video_feats[:,one_map_ds == idx] for idx in groups]
                sizes = [grouped_features[i].shape[1] for i in range(len(grouped_features))]
                max_size = min(sizes)
                grouped_features = [tensor[:, :max_size] for tensor in grouped_features]
                features = torch.stack(grouped_features)
                magnitude = torch.norm(features, p=2, dim = 1)
                features_l2 = (features/(magnitude.unsqueeze(1).repeat(1,fts,1)+1e-5))
                labels = torch.linspace(0,len(sizes)-1,len(sizes)).long()
                supconloss = loss_fn2(features_l2,labels)
                loss = loss + args.supconloss_w * supconloss

            # --- SMOOTHING LOSS ---
            l3 = torch.tensor(0)
            if args.smoothloss_w > 0:
                l3 = (loss_fn3(x)+loss_fn3(y))/current_bs
                loss = loss + args.smoothloss_w * l3

            # Loss between true target (sample2) and recolorized one
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss of every first batch
            train_loss_epoch.append(loss.cpu().detach().numpy().item()) 

        # --- PLOT FEATURES OF X (last batch, first sample) ---
        if args.plot_feats:
            plot_feats(x, fts)
            writer.add_figure('Features', plt.gcf(), epoch)
            plt.close()

        # --- PLOT PCA-3 OF X (last batch) ---
        if args.plot_pca:
            plot_pca(x,fts)
            writer.add_figure('PCA-3', plt.gcf(), epoch)
            plt.close()

        train_loss.append(torch.tensor(train_loss_epoch).mean())
        writer.add_scalar('Train Loss', train_loss[-1], epoch)
        print('Epoch',epoch+1,'- Train loss:', train_loss[-1].item(), 'Supcon loss:', supconloss.item(), 'Sobel loss:', l3.item(), 'Time:', time.time()-t)        
    writer.close()

    # Saving only the encoder
    torch.save(model.state_dict(), './trained-vos-mc1.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)