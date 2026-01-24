import torch.nn as nn
from norm import RMSNorm


# from vizura deepseek lecture
class MTPHEAD(nn.Module):
    def __init__(self,model_dim,vocabulary_size,num_out_heads,num_attn_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_out_heads =num_out_heads
        self.num_attn_heads = num_attn_heads
        self.vocabulary_size = vocabulary_size

        # why do share the weights between embed and unembed( lm head)
        # wight typing tpo reduce the no of parameters  and semantic symentry
        self.lm_head = nn.Linear(model_dim,vocabulary_size,bias= False)
        # TODO weight typing
        self.proj ==nn.ModuleList([nn.Linear(2* model_dim,model_dim) for _ in range(num_out_heads)])
        
        # why transformer encode and why not decoeder?

        self.transformer_layer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_attn_heads, batch_first=True)
            for _ in range(num_out_heads)
        ])
        self.rmsnorm= RMSNorm(model_dim)
    def forward(self,input_ids):
        B,seq_len= input_ids.shape
        device = input_ids.device

        # embed =self.embed(token_ids)
        if init_vec is None:
            hidden_0 =embeds
        else:
            hidden_0 = init_vector
        outputs = []
        for i in range(0,seq_len-num_out_heads):
            hidden_prev = hidden_0[:,i,:]

            logits_mtp=[]
            for head_idx in range(self.num_out_heads):
                # 0+0+1 in first iteration
                future_token_pos = i+(head_idx+1)
                token_embeds = embeds[:,future_token_pos,:]

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
                
        
