import torch
import torch.nn as nn
import torch.nn.functional as F

def infonce_loss(predicted_embeddings, target_embeddings, temperature=0.07):
    """
    Compute InfoNCE (NT-Xent) loss.
    
    Args:
        predicted_embeddings: Tensor of shape [batch_size, embedding_dim]
        target_embeddings: Tensor of shape [batch_size, embedding_dim]
        temperature: Scaling parameter for the logits
        
    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix (cosine similarity)
    logits = (pred_norm @ target_norm.T) / temperature
    
    # Create labels (diagonal elements are positive pairs)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss