import torch

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