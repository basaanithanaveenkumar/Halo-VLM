from models.vit import VisTransformer
from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
import torch
import torch.nn as nn
from models.norm import RMSNorm

from models.positional_embeddings import SinusoidalPositionalEmbedding
class HaloVLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, num_tokens=1,num_future_token=4,num_atten_heads= 32):
        super().__init__()
        self.vis_enc = VisTransformer(img_size=224, p_size=16, in_chans=3, emb_dim=emb_dim, num_layers=6, num_heads=16, mlp_dim=512, drop_fact=0.0)
        self.decoder_transformer = DecoderTransformer(num_layers=16, emb_dim=emb_dim, num_heads=num_atten_heads, mlp_dim=1024, drop_fact=0.0)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(5000, emb_dim)
        self.num_future_token = num_future_token
        self.num_atten_heads = num_atten_heads
        self.lm_head = LMHead(hidden_size=emb_dim, vocab_size=vocab_size)
        # Weight typing
        # TODO Need to check with out weight typing
        # weight typing reducees the no of parameters
        self.lm_head.weight = self.token_emb.weight
        self.image_projector = ImageProjector(vision_dim=emb_dim, llm_dim=emb_dim)
        

        # MTP layers
        self.proj ==nn.ModuleList([nn.Linear(2* model_dim,model_dim) for _ in range(num_out_heads)])
        
        # why transformer encode and why not decoeder?

        self.transformer_layer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_atten_heads, batch_first=True)
            for _ in range(num_future_token)
        ])
        self.rmsnorm= RMSNorm(embed_dim)

    def forward(self,images,input_ids,attention_mask,initial_hidden=None):
        
        B = input_ids.size(0)
        device = input_ids.device
        seq_len = input_ids.size(1)
        img_features = self.vis_enc(images)
        img_proj = self.image_projector(img_features)
        #print(img_proj.shape, "image projection shape")
        B, num_img_tokens, D = img_proj.size()
        num_total_tokens = num_img_tokens + seq_len
        text_embeds = self.token_emb(input_ids)
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
        pos_emb = self.pos_embed(torch.arange(combined_embeds.size(1),device=device)).unsqueeze(0).repeat(B, 1, 1)
        combined_embeds = combined_embeds + pos_emb
        transformer_out=self.decoder_transformer(combined_embeds)
        if initial_hidden is None:
            hidden_0 = text_embeds
        else:
            hidden_0 = initial_hidden
        outputs = []
        for i in range(0,seq_len-self.num_future_token):
            hidden_prev = hidden_0[:,i,:]
            logits_mtp=[]
            for head_idx in range(self.num_future_token):
                future_token_pos = i+(head_idx+1)
                token_embeds = text_embeds[:,future_token_pos,:]
                hidden_norm=self.rmsnorm(hidden_prev)
                e_norm= self.rmsnorm(token_embeds)

                comb_m=torch.cat([hidden_norm,e_norm],dim=1)
                proj_mat=self.prok[head_idx](comb_m)

                x = self.transformer_layer[head_idx](proj_mat.unsqueeze(1))
                hidden_cur=x.squeeze(1)

                logits = self.lm_head(hidden_cur)
                logits_mtp.append(logits)
                # for next iteration
                h_prev = h_curr
            logits_mtp=torch.stack([logits_mtp],dim=1)
            outputs.append(logits_mtp)
        final_out=torch.stack(outputs,dim=1)
        return final_out

        
        # # for the multi token prediction
        # # below code is the implementation from the paper https://arxiv.org/abs/2404.19737
        # if self.num_tokens>1:
        #     final_mtp_logits=[]
        #     transformer_out_d=transformer_out.detach()
        #     transformer_out_d.requires_grad = True
        #     for head_idx in range(self.num_tokens):
        #         logits=self.lm_heads[head_idx](transformer_out_d)
        #         final_mtp_logits.append(logits)
        #     #final_out=self.lm_head(transformer_out)
        #     return final_mtp_logits
        # else:
        #     return self.lm_head(transformer_out)

# write the code to test the forward pass of the model
import torch
import torch.nn as nn

# Test the forward pass
def test_forward_pass():
    # Model hyperparameters
    vocab_size = 32000  # typical tokenizer vocab size
    emb_dim = 512
    batch_size = 2
    seq_len = 77
    img_size = 224
    
    # Initialize model
    model = BasicVLM(vocab_size=vocab_size, emb_dim=emb_dim)
    model.eval()  # Set to evaluation mode
    
    # Create dummy inputs
    dummy_images = torch.randn(batch_size, 3, img_size, img_size)
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Input IDs: {dummy_input_ids.shape}")
    print(f"  Attention mask: {dummy_attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(dummy_images, dummy_input_ids, dummy_attention_mask)
    
    print(f"\nForward pass successful!")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size, num_img_tokens + seq_len, vocab_size)")
    
    # Additional checks
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[-1] == vocab_size, "Vocab size mismatch"
    
    print(f"\nOutput statistics:")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    return True

# Run the test
if __name__ == "__main__":
    test_forward_pass()
