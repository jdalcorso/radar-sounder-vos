import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random

from dataset import SingleVideo, SingleVideoMCORDS1
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from imported.crw import CRW
from sklearn.cluster import KMeans

def dot_product_attention(tensor_A, tensor_B):
    # Normalize
    tensor_A = (tensor_A - tensor_A.mean()) / tensor_A.std()
    tensor_B = (tensor_B - tensor_B.mean()) / tensor_B.std()

    # Reshape tensors to have the same shape (features x N), where N is height * width
    A_flat = tensor_A.view(tensor_A.shape[0], -1)  # Reshape to (features, N)
    B_flat = tensor_B.view(tensor_B.shape[0], -1)  # Reshape to (features, N)

    # Calculate attention scores as the dot product
    attention_scores = torch.matmul(A_flat.t(), B_flat)  # Transpose A_flat and multiply by B_flat
    #print(A_flat.mean(), B_flat.mean(), attention_scores.mean())

    # Apply a softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

    # Compute the attended features using the attention weights
    #attended_features_A = torch.matmul(B_flat, attention_weights.t())  # Multiply B_flat by transposed attention_weights
    #attended_features_A = attended_features_A.view(tensor_A.shape)  # Reshape back to original shape

    #return attended_features_A
    return attention_weights


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def runid(log_folder = './radar_vos/logs'):
    # log_folder must onlu contain folders with name 'run_xxx'
    listdir = os.listdir(log_folder)
    runs = []
    [runs.append(int(run[4:])) for run in listdir]
    if len(runs)>0:
        nrun = max(runs)+1
    else:
        nrun = 0
    return 'run_'+str(nrun)


def positional_encoding(x, d_model):
    pe = torch.zeros(x.size(1), d_model, device = 'cuda')
    position = torch.arange(0, x.size(1), dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).repeat(x.size(0), 1, 1)
    return x + pe


def plot_feats(x, fts):
    x_feats = x[0,...].view(fts,x.shape[-2],x.shape[-1]).detach()
    square = 4
    ix = 1
    plt.figure(figsize=(26,26))
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(x_feats[0+ix-1, :, :].cpu().detach(), cmap='gray') # 192 is random
            ix += 1
    # show the figure
    plt.show()   
    plt.tight_layout() 
    plt.savefig('lfeats.png')

def plot_pca(x,fts):
    _, _, h, w = x.shape
    x = (x-x.mean())/x.std()
    x_feats = x[0,...].view(fts,-1).transpose(0,1)#.view(fts,56,56).detach()
    U, S, V = torch.pca_lowrank(A = x_feats, q = 3) # q=3 to get RGB
    U = U.transpose(0,1).view(3,h,w)
    U = (U - U.min())/(U.max()-U.min())
    plt.imshow(torch.permute(U,[1,2,0]).cpu().detach())
    plt.savefig('pca.png')

def label_prop_val(model, which_data = 0, plot_kmeans = False, writer = None, epoch = None, normalize = None, one_video = None, one_map=None):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Imagenet transformation and dataset according to arguments
    if which_data == 0:
        num_classes = 4
    else:
        num_classes = 3

    # Initialize testing
    model.eval()

    video = one_video
    label = one_map
    video = video.to(device)
    label = label.to(device)

    _,T,H,W = video.shape
    seg = torch.zeros(H,T*W)
    seg[:,:W] = label

    segk = torch.zeros(H,T*W)
    segk[:,:W] = label

    cfg = {
        'CXT_SIZE' : 20, 
        'RADIUS' : 11,
        'TEMP' : 10.00,
        'KNN' : 60,
    }

    # Define label propagation model (from Jabri et al.)
    lp = CRW(cfg)
    feats = []
    masks = []

    for i in range(0,T-1):
        v = video[:,i,:,:].unsqueeze(0).unsqueeze(0).detach()
        sample = v[:,:,0,:,:]
        sample = normalize(v[:,:,0,:,:])    
        
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
        masks.append(ctx)
        feats.append(x)

        mask = lp.forward(feats = feats, lbls = masks)
        mask = mask['masks_pred_idx'].to(device)
        next_lbl = mask
        next_lbl = upscale(next_lbl.unsqueeze(0)).squeeze(0)[-1,...]
        seg[:, W*i:W*i+W] = next_lbl

        if plot_kmeans:
            kmeans = KMeans(3, n_init='auto', random_state=1)
            kmeans_feats = torch.permute(x.squeeze(0).view(nf,-1).cpu().detach(),[1,0])
            kmeans_res = torch.tensor(kmeans.fit(kmeans_feats).labels_).view(x.shape[-2],x.shape[-1])
            kmeans_res = upscale(kmeans_res.unsqueeze(0)).squeeze(0)
            segk[:, W*i:W*i+W] = kmeans_res
            plt.imshow(segk.cpu().detach())
            plt.savefig('lblk.png')
            plt.close()

    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    writer.add_figure('Label prop', plt.gcf(), epoch)
    plt.close()