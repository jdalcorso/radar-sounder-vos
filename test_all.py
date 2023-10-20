import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import random
import matplotlib.pyplot as plt

from dataset import TestDataset
from model import CustomCNN
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix

from imported.crw import CRW

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomCNN()
model.to(device)
num_devices = torch.cuda.device_count()
if num_devices >= 2:
    model = nn.DataParallel(model)
model.load_state_dict(torch.load('./trained-vos-s.pt'))

num_classes = 4
dataset = TestDataset()

# Define cfg for labelprop
cfg = {
        'CXT_SIZE' : 10, 
        'RADIUS' : 5,
        'TEMP' : 10.01,
        'KNN' : 10,
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

        # Append new mask and features, run labelprop
        masks.append(ctx)
        feats.append(x)
        mask = lp.forward(feats = feats, lbls = masks)

        mask = mask['masks_pred_idx'].to(device)
        next_lbl = mask
        next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)[-1,...]
        seg[:, W*i:W*i+W] = next_lbl

        if i%4==0:
            plt.imshow(seg)
            plt.tight_layout()
            plt.savefig('seg.png')
            plt.close()

    seg_list.append(seg)

# Concat seg_list to match the dimension of the full ground truth segmentation
predicted_seg = torch.cat(seg_list, dim = 1).flatten()
gt_seg = dataset.sg.flatten()

# Compute reports
print('Computing reports ...')
print('')
print(classification_report(gt_seg, predicted_seg))
print(confusion_matrix(gt_seg, predicted_seg))
