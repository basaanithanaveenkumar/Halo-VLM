

from models.vit import VisTransformer
from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
import torch
import torch.nn as nn

from vl_jepa.v_jepa import VJePA2Backbone

from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from models.positional_embeddings import SinusoidalPositionalEmbedding
class VL_JEPA_VLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=768):
        super().__init__()
        self.x_encoder = VJePA2Backbone( model_name="facebook/vjepa2-vitl-fpc64-256", H=224, W=224)
        self.y_encoder = DecoderTransformer(num_layers=16, emb_dim=emb_dim, num_heads=32, mlp_dim=1024, drop_fact=0.0)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(5000, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.image_projector = ImageProjector(vision_dim=1024, llm_dim=emb_dim)
        self.target_encoder = SentenceTransformer("google/embeddinggemma-300m")
        self.tokenizer  = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
        
        # TODO need to find a better layer
        # pooling and projection layer
        self.pool_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # → [2, 768, 1]
                nn.Flatten(start_dim=1),  # → [2, 768]
                nn.Linear(emb_dim, emb_dim)       # → [2, 768]
            )

    

    def forward(self,images,input_ids,attention_mask):
        
        query= input_ids[:,:3]
        target= self.target_encoder.encode_query(self.tokenizer.decode(input_ids[:,3:]))
        B = query.size(0)
        device = query.device
        seq_len = query.size(1)
        
        # img_features = self.x_encoder(images)
        batch_features = []
        for i in range(images.shape[0]):  # Iterate batch dim
            single_img = images[i:i+1]    # [1, C, H, W] - keeps batch dim for encoder
            feat = self.x_encoder(single_img)  # Encode one image
            batch_features.append(feat.squeeze(0))  # Remove singleton batch dim if needed

        img_features = torch.stack(batch_features, dim=0)
        img_proj = self.image_projector(img_features)

        #print(img_proj.shape, "image projection shape")
        B, num_img_tokens, D = img_proj.size()
        num_total_tokens = num_img_tokens + seq_len
        query_text_embeds = self.token_emb(query)
        combined_embeds = torch.cat([img_proj, query_text_embeds], dim=1)
        pos_emb = self.pos_embed(torch.arange(combined_embeds.size(1),device=device)).unsqueeze(0).repeat(B, 1, 1)
        combined_embeds = combined_embeds + pos_emb
        transformer_out=self.y_encoder(combined_embeds)
        transformer_out = self.layer_norm(transformer_out)
        transformer_out= self.pool_proj(transformer_out.permute(0,2,1))

        return transformer_out,target

model = VL_JEPA_VLM(vocab_size=30522)  # BERT vocab
model.to(device='cuda:0')
images = torch.randn(2, 224, 224, 3,device='cuda:0')
input_ids = torch.randint(0, 30522, (2, 10),device='cuda:0')  # [B, seq_len]
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    output,target = model(images, input_ids, attention_mask)
print(output.shape)  # Should be [B, total_tokens, vocab_size]
print(target.shape)


import torch
import torch.nn.functional as F
from PIL import Image
import requests

def test_retrieval(model, processor):
    # Get image from URL
    raw_image = Image.open(requests.get(URL, stream=True).raw)
    
    # Process image using processor
    pixel_values = processor.image_processor(raw_image, return_tensors="np").pixel_values
    
    # Convert to torch tensor and reshape to match model input
    # Original expects [B, C, H, W] but we have [B, H, W, C] from numpy
    img = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).to(model.x_encoder.device)
    
    # Tokenize queries
    q_ids = torch.from_numpy(
        processor.tokenizer("Caption: What is this?", return_tensors="np").input_ids
    ).to(model.x_encoder.device)
    
    a_car_ids = torch.from_numpy(
        processor.tokenizer("A car", return_tensors="np").input_ids
    ).to(model.x_encoder.device)
    
    a_noise_ids = torch.from_numpy(
        processor.tokenizer("Noise", return_tensors="np").input_ids
    ).to(model.x_encoder.device)
    
    # Get prediction embedding
    # Note: model expects (images, input_ids, attention_mask)
    with torch.no_grad():
        pred_emb, _ = model(img, q_ids, torch.ones_like(q_ids))  # [B, D]
    
    # Get target embeddings
    # We need to use the target_encoder properly
    with torch.no_grad():
        # Using SentenceTransformer to encode the text
        target_car = torch.from_numpy(
            model.target_encoder.encode("A car", convert_to_tensor=True).cpu().numpy()
        ).unsqueeze(0).to(model.x_encoder.device)
        
        target_noise = torch.from_numpy(
            model.target_encoder.encode("Noise", convert_to_tensor=True).cpu().numpy()
        ).unsqueeze(0).to(model.x_encoder.device)
    
    # Normalize embeddings
    pred_norm = F.normalize(pred_emb, p=2, dim=-1)
    car_norm = F.normalize(target_car, p=2, dim=-1)
    noise_norm = F.normalize(target_noise, p=2, dim=-1)
    
    # Compute similarity scores
    score_car = (pred_norm @ car_norm.T).item()
    score_noise = (pred_norm @ noise_norm.T).item()
    
    print(f"Similarity to 'A car': {score_car:.4f}")
    print(f"Similarity to 'Noise': {score_noise:.4f}")

    # Optional: Return a boolean indicating if retrieval is successful
    return score_car > score_noise
