import torch.nn as nn

class ImageProjector(nn.Module):
    def __init__(self, vision_dim=512, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(vision_dim, llm_dim // 4),
                nn.LayerNorm(llm_dim // 4),
                nn.GELU(),
                nn.Linear(llm_dim//4, llm_dim // 2),
                nn.LayerNorm(llm_dim // 2),
                nn.GELU(),
                nn.Linear(llm_dim//2,llm_dim )
            )

    def forward(self, img_features):
        return self.proj(img_features)