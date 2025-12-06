import torch
import timm


import timm

# List all DINOv3 models
dinov3_models = [m for m in timm.list_models() if 'dinov3' in m.lower()]
print(dinov3_models)

# Load smallest DINOv3 (ViT-Small with patch size 14)
model = timm.create_model(
    'vit_small_patch14_dinov3.lvd1689m',
    pretrained=True,
    features_only=False  # Set to True if you want intermediate features
)

# Move to GPU
model = model.to('cuda')
model.eval()

# Feature extraction
batch_size = 8
x = torch.randn(batch_size, 3, 224, 224).cuda()

with torch.no_grad():
    features = model(x)

print(f"Input shape: {x.shape}")
print(f"Output features shape: {features.shape}")
print(f"Feature dimension: {features.shape[-1]}")  # 384 for ViT-S
