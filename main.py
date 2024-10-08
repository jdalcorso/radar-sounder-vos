import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import time

from dataset import VideoDataset, SingleVideo, MCORDS1Dataset, SingleVideoMCORDS1, MCORDS1Miguel
from model import CustomCNN, CustomCNN2, CustomCNN3
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from utils import dot_product_attention, SupConLoss, runid, positional_encoding, plot_feats, plot_pca, label_prop_val

seed = 123  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--image_size', default=(400,48), type=int) # Change this, if you change args.which_data
    parser.add_argument('--which_data', default = 0, type=int, help = '0 for MCORDS1_2010, 1 for MCORDS3 ds')
    # Loss parameters
    parser.add_argument('--huber', default = False, type = bool)
    parser.add_argument('--supconloss_w', default=0.0, type=float)
    parser.add_argument('--l2regloss_w', default=0.000, type=float)
    # Training parameters
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1E-4, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    # Plots and folders
    parser.add_argument('--pos_encode', default = False, type = bool)
    parser.add_argument('--plot_feats', default = True, type = bool)
    parser.add_argument('--plot_pca', default = True, type = bool)
    parser.add_argument('--validation', default = True, type = bool)
    parser.add_argument('--datafolder',default='/data/videos/class_0')
    parser.add_argument('--savefolder',default='./radar_vos_run/')
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_id = runid()
    writer = SummaryWriter('./radar_vos/logs/'+run_id)
    writer.add_text('arguments',str(args))

    model = CustomCNN()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)

    # Choose dataset, Imagenet transformation and single reference video according to arguments
    if args.which_data == 0:
        dataset = MCORDS1Dataset(dim = args.image_size , factor = 1)
        normalize = transforms.Normalize(mean = [0.0], std = [1.0])
        one_video = SingleVideoMCORDS1(dim = args.image_size)
        ov, one_map = one_video[0]
        one_video = ov[0,0,:,:].unsqueeze(0).unsqueeze(0)
        one_video = normalize(one_video)
  
    if args.which_data == 1:
        dataset = VideoDataset('/data/videos/class_0')
        normalize = transforms.Normalize(mean = [-458.0144], std = [56.2792])
        one_video = SingleVideo()
        ov, one_map = one_video[0]
        one_video = ov[0,0,:,:].unsqueeze(0).unsqueeze(0)
        one_video = normalize(one_video)

    if args.which_data == 2:
        dataset = MCORDS1Miguel(dim = (1248,24))
        normalize = transforms.Normalize(mean = [0.0], std = [1.0])
        one_video = SingleVideoMCORDS1(dim = args.image_size)
        ov, one_map = one_video[0]
        one_video = ov[0,0,:,:].unsqueeze(0).unsqueeze(0)
        one_video = normalize(one_video)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    # Initialize optimizer
    optimizer = AdamW(params=model.parameters() ,lr = args.lr, betas=(0.9, 0.95))
    if args.huber:
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.MSELoss()
    loss_fn2 = SupConLoss()

    # Initialize training
    print('Training on:', device)
    if device.type == 'cuda':
        print('Total CUDA memory:',torch.cuda.get_device_properties(0).total_memory/(1024**3))
        print('Reserved CUDA memory:',torch.cuda.memory_reserved(0)/(1024**3))
        print('Allocated CUDA memory:',torch.cuda.memory_allocated(0)/(1024**3))

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
            sample1 = samples[:,:,0,:,:]
            sample2 = samples[:,:,1,:,:]
            sample1 = normalize(sample1)
            sample2 = normalize(sample2)

            current_bs = samples.shape[0]

            plt.imshow(sample1.cpu().detach()[0,0,...])
            plt.savefig('sample.png')
            plt.close()

            samples = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
            x = model(sample1)
            y = model(sample2)

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

            # --- L2-REGULARIZATION LOSS ---
            regloss = torch.tensor(0, device = device, dtype = torch.float)
            if args.l2regloss_w > 0:
                for weight in model.parameters():
                    regloss += torch.sum(torch.square(weight))
                loss = loss + args.l2regloss_w * regloss

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

        # --- VALIDATION: LABEL PROP ---
        if args.validation:
            label_prop_val(model = model, which_data = 0, plot_kmeans = False, writer = writer, epoch = epoch, normalize = normalize, one_video = ov, one_map=one_map)

        print('Epoch',epoch+1,'- Train loss:', train_loss[-1].item(), 'Supcon loss:', supconloss.item(), 'L2 loss:', regloss.item(), 'Time:', time.time()-t)        
    writer.close()

    # Saving only the encoder
    torch.save(model.state_dict(), './trained-vos.pt')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)