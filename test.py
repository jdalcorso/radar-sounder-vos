import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import random
import matplotlib.pyplot as plt

from dataset import SingleVideo, SingleVideoMCORDS1
from model import CustomCNN
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from imported.crw import CRW
from sklearn.cluster import KMeans

seed = 123  
torch.manual_seed(seed)
random.seed(seed)

def get_args_parser():
    # Default are from MAE (He et al. 2021)
    parser = argparse.ArgumentParser('VOS test', add_help=False)
    parser.add_argument('--which_data', default = 0, type=int, help = '0 for MCORDS1_2010, 1 for Miguel ds')
    # Label-prop parameters
    # Loss parameters
    # Test parameters
    parser.add_argument('--plot_kmeans', default = False, type=bool)
    parser.add_argument('--append_color', default = True, type = bool)
    parser.add_argument('--append_posv', default = True, type = bool)
    return parser

def main(args):

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomCNN()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./trained-vos-latest.pt'))

    # Imagenet transformation and dataset according to arguments
    if args.which_data == 0:
        num_classes = 4
        dataset = SingleVideoMCORDS1()
        normalize = transforms.Normalize(mean = [0.0], std = [1.0])

    else:
        num_classes = 3
        dataset = SingleVideo('/data/videos/class_0')
        normalize = transforms.Normalize(mean = [-458.0144], std = [56.2792])

    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    # Initialize training
    print('Testing on:', device)
    if device.type == 'cuda':
        print('Total CUDA memory:',torch.cuda.get_device_properties(0).total_memory/1024**3)
        print('Reserved CUDA memory:',torch.cuda.memory_reserved(0)/1024**3)
        print('Allocated CUDA memory:',torch.cuda.memory_allocated(0)/1024**3)

    print('---------------------------------------------------------')
    print('--- START TESTING ---')
    model.eval()

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
        'RADIUS' : 10,
        'TEMP' : 10.01,
        'KNN' : 10,
    }

    # Define label propagation model (from Jabri et al.)
    lp = CRW(cfg, verbose = True)
    feats = []
    masks = []

    for i in range(0,T-1):
        print('Range-line:',i)
        v = video[:,i,:,:].unsqueeze(0).unsqueeze(0).detach()
        sample = v[:,:,0,:,:]
        sample = normalize(v[:,:,0,:,:])

        #v = torch.cat([sample1.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
        with torch.inference_mode():
            x = model(sample)
        x = (x - x.mean()) / x.std()

        _, nf, fH, fW = x.shape 

        downscale = transforms.Resize((fH,fW), interpolation=InterpolationMode.NEAREST)
        upscale = transforms.Resize((H,W), interpolation=InterpolationMode.NEAREST)

        # downscale label and turn into a mask
        if i !=0:
            label = next_lbl
        label = downscale(label.unsqueeze(0)).squeeze(0)

        ctx = torch.zeros(num_classes, fH, fW, device = device)
        for class_idx in range(0, num_classes):
            mask = (label == class_idx).unsqueeze(0).float()
            ctx[class_idx, :, :] = mask
        ctx = ctx.unsqueeze(0)

        # Add posenc
        if args.append_posv:
            p = torch.linspace(-1,1,100).unsqueeze(0).transpose(0,1)
            p = p.repeat(1,12).unsqueeze(0).unsqueeze(0).cuda()
            x = torch.cat([x,p],dim = 1)

        # Add color
        if args.append_color:
            sample = downscale(sample)
            x = torch.cat([x,sample],dim = 1)


        masks.append(ctx)
        feats.append(x)
        mask = lp.forward(feats = feats, lbls = masks)

        plt.imshow(mask['masks_pred_idx'][-1,:,:])
        plt.savefig('lconf.png')
        plt.close()

        mask = mask['masks_pred_idx'].to(device)
        next_lbl = mask
        next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)[-1,...]
        seg[:, W*i:W*i+W] = next_lbl

        if args.plot_kmeans:
            kmeans = KMeans(3, n_init='auto', random_state=1)
            kmeans_feats = torch.permute(x.squeeze(0).view(nf,-1).cpu().detach(),[1,0])
            kmeans_res = torch.tensor(kmeans.fit(kmeans_feats).labels_).view(x.shape[-2],x.shape[-1])
            kmeans_res = upscale(kmeans_res.unsqueeze(0)).squeeze(0)
            segk[:, W*i:W*i+W] = kmeans_res
            plt.imshow(segk.cpu().detach())
            plt.savefig('lblk.png')
            plt.close()

    print('--- TEST DONE ---')
    print('Saving results in lbl.png')
    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    plt.close()

    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)