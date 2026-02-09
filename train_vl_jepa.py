"""
Minimal Training Example for VQA with InfoNCE Loss

This is a simplified version showing the core training loop.
Adapt this to your specific model architecture.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

from jepa_dataloader import create_dataloader

from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
import torch
import torch.nn as nn

from models.vit import VisTransformer
from vl_jepa.v_jepa import VJePA2Backbone

from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


from models.positional_embeddings import SinusoidalPositionalEmbedding
from vl_jepa_visualizer import TrainingVisualizer, visualize_batch



def detailed_parameter_analysis(model):
    """
    Detailed analysis of parameters by module and layer type.
    """
    print("="*100)
    print(f"{'MODULE HIERARCHY':<60} {'PARAMS':>15} {'%':>8} {'TRAINABLE':>10}")
    print("-"*100)
    
    total_params = 0
    total_trainable = 0
    module_stats = {}
    
    # Collect statistics
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if params > 0:
            module_type = module.__class__.__name__
            module_stats.setdefault(module_type, {"count": 0, "params": 0, "trainable": 0})
            module_stats[module_type]["count"] += 1
            module_stats[module_type]["params"] += params
            module_stats[module_type]["trainable"] += trainable_params
            
            total_params += params
            total_trainable += trainable_params
            
            # Print module info
            pct = (params / total_params * 100) if total_params > 0 else 0
            trainable_pct = (trainable_params / params * 100) if params > 0 else 0
            
            print(f"{name:<60} {params:>15,} {pct:>7.1f}% {trainable_pct:>9.1f}%")
    
    print("-"*100)
    
    # Summary by module type
    print("\n" + "="*100)
    print(f"{'MODULE TYPE':<25} {'COUNT':>8} {'PARAMS':>15} {'%':>8} {'TRAINABLE':>12}")
    print("-"*100)
    
    for module_type, stats in sorted(module_stats.items(), key=lambda x: x[1]["params"], reverse=True):
        pct_total = (stats["params"] / total_params * 100) if total_params > 0 else 0
        trainable_pct = (stats["trainable"] / stats["params"] * 100) if stats["params"] > 0 else 0
        
        print(f"{module_type:<25} {stats['count']:>8,} {stats['params']:>15,} "
              f"{pct_total:>7.1f}% {trainable_pct:>11.1f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<25} {'-':>8} {total_params:>15,} {100:>7.1f}% {'-':>12}")
    print(f"{'TRAINABLE':<25} {'-':>8} {total_trainable:>15,} "
          f"{(total_trainable/total_params*100):>7.1f}% {'-':>12}")
    print("="*100)
    
    return total_params, total_trainable


class PoolProj(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Learnable query vector for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        # Linear projection for attention scores
        self.attn_score = nn.Linear(emb_dim, 1)
        
        # Optional projection after pooling
        self.proj = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):
        # x: [B, S, emb_dim]
        
        # Method 1: Simple attention pooling
        # attn_weights = F.softmax(self.attn_score(x), dim=1)  # [B, S, 1]
        # pooled = torch.sum(attn_weights * x, dim=1)  # [B, emb_dim]
        
        # Method 2: Multi-query attention pooling (more expressive)
        attn_scores = torch.matmul(self.query, x.transpose(1, 2))  # [B, 1, S]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, S]
        pooled = torch.matmul(attn_weights, x).squeeze(1)  # [B, emb_dim]
        
        return self.proj(pooled)


class VL_JEPA_VLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=768):
        super().__init__()
        self.x_encoder = VJePA2Backbone( model_name="facebook/vjepa2-vitl-fpc64-256", H=224, W=224)
        self.y_encoder = DecoderTransformer(num_layers=12, emb_dim=emb_dim, num_heads=16, mlp_dim=812, drop_fact=0.0)
        
        self.pos_embed = nn.Embedding(5000, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.image_projector = ImageProjector(vision_dim=1024, llm_dim=emb_dim)
        
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # TODO need to find a better layer
        # pooling and projection layer
        self.pool_proj = PoolProj(emb_dim)

    

    def forward(self,images,query,target):
        
        
       
        B = query.size(0)
        device = query.device
        seq_len = query.size(1)
        print(query)
        # img_features = self.x_encoder(images)
        batch_features = []
        for i in  images:  # Iterate batch dim
            #single_img = images[i:i+1]    # [1, C, H, W] - keeps batch dim for encoder
            print(i.shape, "input_images shape")
            feat = self.x_encoder(i)  # Encode one image
            print(feat.shape)
            batch_features.append(feat.squeeze(0))  # Remove singleton batch dim if needed

        img_features = torch.stack(batch_features, dim=0)
        img_proj = self.image_projector(img_features.to('cuda'))

        #print(img_proj.shape, "image projection shape")
        B, num_img_tokens, D = img_proj.size()
        num_total_tokens = num_img_tokens + seq_len
        query_text_embeds = self.token_emb(query)
        combined_embeds = torch.cat([img_proj, query_text_embeds], dim=1)
        pos_emb = self.pos_embed(torch.arange(combined_embeds.size(1),device=device)).unsqueeze(0).repeat(B, 1, 1)
        combined_embeds = combined_embeds + pos_emb
        transformer_out=self.y_encoder(combined_embeds)
        transformer_out = self.layer_norm(transformer_out)
        transformer_out= self.pool_proj(transformer_out)#.permute(0,2,1))

        return transformer_out,target
 

# ============================================================================
# Loss Function
# ============================================================================

# def infonce_loss(predicted_embeddings, target_embeddings, temperature=1.0):
#     """
#     Compute InfoNCE (NT-Xent) loss.
#     """
#     # Normalize embeddings
#     pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
#     target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    
#     # Compute similarity matrix
#     logits = (pred_norm @ target_norm.T) / temperature
    
#     # Create labels
#     batch_size = logits.shape[0]
#     labels = torch.arange(batch_size, device=logits.device)
    
#     # Compute loss
#     loss = F.cross_entropy(logits, labels)
def are_embeddings_same(predicted_embeddings, target_embeddings, threshold=1e-6):
    """
    Check if two embedding tensors are identical (or very similar).
    Returns True if they're effectively the same.
    """
    if predicted_embeddings.shape != target_embeddings.shape:
        return False
    
    # Check exact equality
    if torch.equal(predicted_embeddings, target_embeddings):
        print("WARNING: Embeddings are EXACTLY identical (torch.equal returns True)")
        return True
    
    # Check for very close similarity
    diff = torch.abs(predicted_embeddings - target_embeddings)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if max_diff < threshold:
        print(f"WARNING: Embeddings are nearly identical (max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e})")
        return True
    
    # Check correlation
    if predicted_embeddings.numel() > 1:
        pred_flat = predicted_embeddings.flatten()
        target_flat = target_embeddings.flatten()
        correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
        print(f"Correlation between embeddings: {correlation:.6f}")
        if correlation > 0.99:  # Very high correlation
            print(f"WARNING: Embeddings have very high correlation: {correlation:.6f}")
            return True
    
    return False
def infonce_loss(predicted_embeddings, target_embeddings, temperature=0.07, debug=True):
    """
    Compute InfoNCE loss with comprehensive debugging.
    """
    # First, check if embeddings are the same
    same_check = are_embeddings_same(predicted_embeddings, target_embeddings)
    
    if same_check:
        print("\n" + "="*60)
        print("CRITICAL WARNING: Predicted and target embeddings are identical!")
        print("This will cause the loss to approach zero.")
        print("Possible causes:")
        print("1. Model is outputting the same embeddings regardless of input")
        print("2. Target embeddings are being incorrectly set")
        print("3. Gradient issues or collapsed representations")
        print("="*60 + "\n")
        
        # Force a high loss to prevent gradient issues
        # or return a warning value
        batch_size = predicted_embeddings.shape[0]
        warning_loss = torch.tensor([math.log(batch_size)], 
                                   device=predicted_embeddings.device)
        return warning_loss
    
    # Normalize embeddings
    pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    
    if debug:
        print(f"\nEmbedding Statistics:")
        print(f"Predicted - min: {predicted_embeddings.min():.4f}, max: {predicted_embeddings.max():.4f}, "
              f"mean: {predicted_embeddings.mean():.4f}, std: {predicted_embeddings.std():.4f}")
        print(f"Target    - min: {target_embeddings.min():.4f}, max: {target_embeddings.max():.4f}, "
              f"mean: {target_embeddings.mean():.4f}, std: {target_embeddings.std():.4f}")
        
        # Check if all embeddings are zero
        if torch.allclose(predicted_embeddings, torch.zeros_like(predicted_embeddings), atol=1e-6):
            print("WARNING: Predicted embeddings are all zeros!")
        if torch.allclose(target_embeddings, torch.zeros_like(target_embeddings), atol=1e-6):
            print("WARNING: Target embeddings are all zeros!")
    
    # Compute similarity matrix
    similarity = pred_norm @ target_norm.T
    logits = similarity / temperature
    
    if debug:
        print(f"\nSimilarity Matrix Diagnostics:")
        print(f"Similarity shape: {similarity.shape}")
        print(f"Diagonal (positives): {torch.diag(similarity)[:5].tolist()}")
        print(f"Diagonal stats - min: {torch.diag(similarity).min():.4f}, "
              f"max: {torch.diag(similarity).max():.4f}, "
              f"mean: {torch.diag(similarity).mean():.4f}")
        print(f"Off-diagonal mean: {similarity[~torch.eye(similarity.shape[0], dtype=bool)].mean():.4f}")
        print(f"Temperature: {temperature}")
        print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Create labels
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute loss
    loss = F.cross_entropy(logits, labels)
    
    if debug:
        print(f"\nLoss Computation:")
        print(f"Batch size: {batch_size}")
        print(f"Loss: {loss.item():.6f}")
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
            print(f"Accuracy: {accuracy:.4f}")
            
            # If accuracy is 1.0, loss should be very small
            if accuracy > 0.99:
                print("WARNING: Accuracy > 99%, loss will be near zero!")
    
    return loss
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    return total_params, trainable_params, non_trainable_params

# ============================================================================
# Minimal Training Example
# ============================================================================

def collate_fn_pad_images(batch):
    """Pad images to same number in batch."""
    images_list, questions, answers = [], [], []
    
    for img_tensor, question, answer in batch:
        images_list.append(img_tensor)
        questions.append(question)
        answers.append(answer)
    
    # Find max number of images
    max_num_images = max(img.shape[0] for img in images_list)
    
    # Pad all to max
    padded_images = []
    for imgs in images_list:
        if imgs.shape[0] < max_num_images:
            padding = torch.zeros(
                max_num_images - imgs.shape[0],
                imgs.shape[1], imgs.shape[2], imgs.shape[3]
            )
            imgs = torch.cat([imgs, padding], dim=0)
        padded_images.append(imgs)
    
    images_batch = torch.stack(padded_images)  # [B, N, C, H, W]
    
    return images_batch, questions, answers


class VisualizationConfig:
    """Configuration for visualization."""
    
    # Visualization settings
    visualize_every = 100  # Visualize every N steps
    output_dir = "./visualizations"
    fps = 1  # Frames per second for video
    max_samples_to_show = 6  # Max samples per frame
    max_images_per_sample = 4  # Max images per sample
    
    # Video settings
    create_video_after_training = True
    cleanup_frames = False  # Keep frames after video creation
    video_name = "training_visualization.mp4"

def train_minimal_example():
    """
    Minimal training loop.
    
    REPLACE YourModel with your actual model class!
    """
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 12
    num_epochs = 280
    learning_rate = 1e-4
    temperature = 0.07
    visualizer = TrainingVisualizer(
        output_dir=VisualizationConfig().output_dir,
        fps=VisualizationConfig().fps,
        max_samples_to_show=VisualizationConfig().max_samples_to_show,
        max_images_per_sample=VisualizationConfig().max_images_per_sample
    )
    target_encoder = SentenceTransformer("google/embeddinggemma-300m")
    tokenizer  = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Create dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    train_loader = create_dataloader(
        split='train',
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        transform=transform,
    )
    
    
    model = VL_JEPA_VLM(vocab_size=vocab_size,emb_dim=768).to('cuda')
    for param in model.x_encoder.parameters():
        param.requires_grad = False
    count_parameters(model)
    detailed_parameter_analysis(model)
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training on {device} for {num_epochs} epochs...")
    print("="*80)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        global_step=0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(pbar):
            # Move data to device
            images = batch['images']  # [B, max_num_images, C, H, W]
            questions = batch['questions']
            answers = batch['answers']
            # Move to device
            # Forward pass - YOUR MODEL SHOULD RETURN THESE!
            query= tokenizer(questions,
                                max_length= 50,  # Reserve 1 space for EOS token
                                padding='max_length',
                                truncation=True,
                                return_tensors="pt" )["input_ids"].to("cuda")
            #target= torch.from_numpy(target_encoder.encode_query(answers)).to('cuda')
            encoded_list = []

            for ans in answers:
                encoded_ans = target_encoder.encode_query([ans])  # Tokenize one at a time; adjust if returns list vs np
                if isinstance(encoded_ans, list):
                    encoded_ans = torch.tensor(encoded_ans, dtype=torch.long)  # Handle list → tensor
                else:  # Assume np.ndarray
                    encoded_ans = torch.from_numpy(encoded_ans)
                encoded_list.append(encoded_ans)
            target = torch.stack(encoded_list,dim=0).squeeze(1).to('cuda') 
            predicted_embeddings, target_embeddings = model(images, query, target)
            print("\nCreating individual videos for each batch sample...")
            # Create results directory
            results_dir = "./video_embedding_visualizations"
            import os
            from datetime import datetime
            # Create results directory
            results_dir = "./video_embedding_visualizations"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # # Option 2: Video animation
            # print("\n2. Creating video animation...")
            # from viz_test import create_video_embedding_video
            # ani_seq = create_video_embedding_video(
            #     target_embeddings, predicted_embeddings, images, answers,
            #     use_pca=True,
            #     save_path=f"{results_dir}/video_animation_{timestamp}.gif",
            #     fps_per_video=1,
            #     figsize=(14, 7)
            # )
            
            # print(f"\nCreated {len(individual_videos)} individual videos:")
            # for i, video_path in enumerate(individual_videos):
            #     print(f"  Sample {i}: {os.path.basename(video_path)}")
            
            # Compute loss
            loss = infonce_loss(predicted_embeddings, target_embeddings, temperature)
            print(loss, "loss in each iteration")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            # Visualize
            if global_step % VisualizationConfig().visualize_every == 0:
                print(f"\n[Step {global_step}] Creating visualization frame...")
                
                # Create visualization (detach tensors for visualization)
                with torch.no_grad():
                    # Option 2: Video animation
                    print("\n2. Creating video animation...")
                    from viz_test import create_video_embedding_video
                    ani_seq = create_video_embedding_video(
                        target_embeddings, predicted_embeddings, images, answers,
                        use_pca=True,
                        save_path=f"{results_dir}/video_animation_{timestamp}.gif",
                        fps_per_video=1,
                        figsize=(14, 7)
                    )
            
            global_step += 1
        
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        print("="*80)
        


if __name__ == "__main__":
    train_minimal_example()