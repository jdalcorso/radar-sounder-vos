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