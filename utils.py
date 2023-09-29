import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random

from dataset import SingleVideo, SingleVideoMCORDS1
from model import RGVOS
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

def combine_masks_to_segmentation(masks):
    """
    Combines binary masks for each class into a segmentation map.

    Args:
        masks (Tensor): Tensor of masks with dimensions [batch_size, num_classes, H, W],
                        where each channel contains a binary mask for a class.

    Returns:
        segmentation_map (Tensor): Combined segmentation map with dimensions [batch_size, H, W],
                                   where each pixel contains the class label (1, 2, ..., num_classes).
    """
    # Get the number of classes and batch size
    batch_size, num_classes, H, W = masks.size()
    
    # Initialize an empty tensor for the segmentation map
    segmentation_map = torch.zeros(batch_size, H, W, device = 'cuda')
    
    # Iterate over each class and assign class labels to pixels in the segmentation map
    for class_idx in range(num_classes):
        class_mask = masks[:, class_idx, :, :]
        segmentation_map += class_mask * (class_idx)  # Class labels start from 1
    
    return segmentation_map


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
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
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
    nrun = max(runs)+1
    return 'run_'+str(nrun)

def positional_encoding(x, d_model):
    """Computes the positional encoding for a given input.

    Args:
        x: A tensor of shape [batch_size, sequence_length, hidden_size].
        d_model: The hidden size of the model.

    Returns:
        A tensor of shape [batch_size, sequence_length, hidden_size] containing the positional encoding.
    """

    pe = torch.zeros(x.size(1), d_model, device = 'cuda')
    position = torch.arange(0, x.size(1), dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).repeat(x.size(0), 1, 1)
    return x + pe

def plot_feats(x, fts):
    x_feats = x[0,...].view(fts,x.shape[-2],x.shape[-1]).detach()
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
            plt.imshow(x_feats[0+ix-1, :, :].cpu().detach(), cmap='gray') # 192 is random
            ix += 1
    # show the figure
    plt.show()    
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
    pass


class SobelSmoothingLoss(nn.Module):
  def __init__(self):
    super(SobelSmoothingLoss, self).__init__()

    # Define the Sobel filter kernels
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device = 'cuda')
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device = 'cuda')

    # Convert the Sobel filter kernels to a PyTorch convolution kernel
    self.sobel_x = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding=1, bias=False, groups=1, device = 'cuda')
    self.sobel_y = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding=1, bias=False, groups=1, device = 'cuda')

    # Set the weights of the convolution kernels to the Sobel filter kernels
    self.sobel_x.weight.data = sobel_x.unsqueeze(0).unsqueeze(0).repeat([512,512,3,3])
    self.sobel_y.weight.data = sobel_y.unsqueeze(0).unsqueeze(0).repeat([512,512,3,3])
    #self.sobel_x.weight.data = sobel_x.unsqueeze(0).unsqueeze(0)
    #self.sobel_y.weight.data = sobel_y.unsqueeze(0).unsqueeze(0)

    self.sobel_x.weight.requires_grad = False
    self.sobel_y.weight.requires_grad = False

  def forward(self, x):
    # Compute the Sobel gradients of the input image
    sobel_x = self.sobel_x(x)
    sobel_y = self.sobel_y(x)

    # Compute the magnitude of the Sobel gradients
    sobel_mag = torch.sqrt(sobel_x**2 + sobel_y**2)

    # Compute the smoothing loss
    smoothing_loss = F.l1_loss(sobel_mag, torch.zeros_like(sobel_mag))

    return smoothing_loss


def label_prop_val(model, which_data = 0, plot_kmeans = False, writer = None, epoch = None):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Imagenet transformation and dataset according to arguments
    if which_data == 0:
        num_classes = 4
        dataset = SingleVideoMCORDS1()
        normalize = transforms.Normalize(mean = [0.0,0.0,0.0], std = [1.0,1.0,1.0])
    else:
        num_classes = 3
        dataset = SingleVideo('/data/videos/class_0')
        normalize = transforms.Normalize(mean = [-458.0144, -458.0144, -458.0144], std = [56.2792, 56.2792, 56.2792])

    resize2resnet = transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST)

    # Initialize testing
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
        'TEMP' : 0.01,
        'KNN' : 10,
    }

    # Define label propagation model (from Jabri et al.)
    lp = CRW(cfg)
    feats = []
    masks = []

    for i in range(1,T-1):
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

        if plot_kmeans:
            kmeans = KMeans(3, n_init='auto', random_state=1)
            kmeans_feats = torch.permute(y.squeeze(0).view(512,-1).cpu().detach(),[1,0])
            kmeans_res = torch.tensor(kmeans.fit(kmeans_feats).labels_).view(56,56)
            kmeans_res = upscale(kmeans_res.unsqueeze(0)).squeeze(0)
            segk[:, W*i:W*i+W] = kmeans_res            
            plt.imshow(segk.cpu().detach())
            plt.savefig('lblk.png')
            plt.close()

    plt.imshow(seg.cpu().detach())
    plt.savefig('lbl.png')
    writer.add_figure('Label prop', plt.gcf(), epoch)
    plt.close()