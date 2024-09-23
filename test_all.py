import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import time

from dataset import TestDataset
from model import CustomCNN, CustomCNN2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix

from imported.crw import CRW

def get_args_parser():
    parser = argparse.ArgumentParser('VOS test', add_help=False)
    # Labelprop parameters
    parser.add_argument('--cxt_size', '-c', default=20, type=int) # Change this, if you change args.which_data
    parser.add_argument('--radius', '-r', default=11, type=int)
    parser.add_argument('--temp', '-t', default=10.0, type=float)
    parser.add_argument('--knn', '-k', default=60, type=int)
    # Test parameters
    parser.add_argument('--remove_unc', default = True, type=bool)
    parser.add_argument('--append_posv', default = False, type=bool)
    parser.add_argument('--append_color', default = False, type=bool)
    # Cases
    parser.add_argument('--color_only', default = False, type=bool)
    return parser

def main(args):
    tim = time.time()
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomCNN2()
    model.to(device)
    num_devices = torch.cuda.device_count()
    if num_devices >= 2:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./trained-vos-latest.pt'))

    num_classes = 4
    dataset = TestDataset()

    # Define cfg for labelprop
    cfg = {
            'CXT_SIZE' : args.cxt_size, 
            'RADIUS' : args.radius,
            'TEMP' : args.temp,
            'KNN' : args.knn,
        }
    print(cfg)

    seg_list = []
    #for rg, sg in dataset:
    for t in range(dataset.nrg):
        rg, sg = dataset[t]
        # Define label prop algorithm
        lp = CRW(cfg, verbose = False)
        feats = []
        masks = []

        # Define reference label
        H,W = dataset.dim
        T = dataset.npatches
        label = sg[:,:W]

        # Initialize predicted segmentation
        seg = torch.zeros(H,W*T)

        for i in range(T-1):
            sample = rg[:,i*W:i*W+W]

            # Compute features, turn label into mask
            with torch.inference_mode():
                    x = model(sample.unsqueeze(0).unsqueeze(0))
            x = (x - x.mean()) / x.std()
            _, nf, fH, fW = x.shape 

            downscale = transforms.Resize((fH,fW), interpolation=InterpolationMode.NEAREST)
            upscale = transforms.Resize((H,W), interpolation=InterpolationMode.NEAREST)

            # Downscale label
            if i !=0:
                label = next_lbl
            label = downscale(label.unsqueeze(0)).squeeze(0)

            # Turn label into a mask
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
                sample = downscale(sample.unsqueeze(0)).cuda().unsqueeze(0)
                x = torch.cat([x,sample],dim = 1)
                if args.color_only:
                    x = x[:,-1:,:,:]

            # Append new mask and features, run labelprop
            masks.append(ctx)
            feats.append(x)
            mask = lp.forward(feats = feats, lbls = masks)

            mask = mask['masks_pred_idx'].to(device)
            next_lbl = mask
            next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)[-1,...]
            seg[:, W*i:W*i+W] = next_lbl

            if i%24==0:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                ax1.imshow(seg,interpolation="nearest")
                ax2.imshow(sg, interpolation="nearest")
                ax3.imshow(rg)
                plt.tight_layout()
                plt.savefig('./radar_vos/output/'+str(t)+'seg.pdf', format = 'pdf', dpi = 100, bbox_inches='tight')
                plt.close()

        seg_list.append(seg)

    # Concat seg_list to match the dimension of the full ground truth segmentation
    predicted_seg = torch.cat(seg_list, dim = 1).flatten()
    gt_seg = dataset.sr.flatten()

    # Remove class 4 (uncertain)
    if args.remove_unc:
        mask = gt_seg != 4
        gt = gt_seg[mask]
        pred = predicted_seg[mask]
    else:
        gt = gt_seg
        pred = predicted_seg

    # Compute reports
    print('Computing reports ...')
    print('')
    print(classification_report(gt, pred))
    print(confusion_matrix(gt, pred))
    print('Time elapsed:',time.time()-tim)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
