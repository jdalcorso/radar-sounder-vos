import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import random
import matplotlib.pyplot as plt

from dataset import SingleVideo, SingleVideoMCORDS1
from model import RGVOS
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
    # Model parameters
    # Loss parameters
    # Test parameters
    parser.add_argument('--plot_kmeans', default = True, type=bool)
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

    # Imagenet transformation and dataset according to arguments
    if args.which_data == 0:
        num_classes = 4
        dataset = SingleVideoMCORDS1()
        normalize = transforms.Normalize(mean = [0.0,0.0,0.0], std = [1.0,1.0,1.0])
    else:
        num_classes = 3
        dataset = SingleVideo('/data/videos/class_0')
        normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792])

    resize2resnet = transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST)

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
        'CXT_SIZE' : 1, 
        'RADIUS' : 10,
        'TEMP' : 100.01,
        'KNN' : 10,
    }

    # Define label propagation model (from Jabri et al.)
    lp = CRW(cfg, verbose = True)
    feats = []
    masks = []

    for i in range(1,T-1):
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

        mask = lp.forward(feats = feats, lbls = masks)

        mask = mask['masks_pred_idx'].to(device)
        next_lbl = mask
        next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)[-1,...]
        seg[:, W*i:W*i+W] = next_lbl

        if args.plot_kmeans:
            kmeans = KMeans(3, n_init='auto', random_state=1)
            kmeans_feats = torch.permute(y.squeeze(0).view(512,-1).cpu().detach(),[1,0])
            kmeans_res = torch.tensor(kmeans.fit(kmeans_feats).labels_).view(56,56)
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