import torch
import torch.nn as nn

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