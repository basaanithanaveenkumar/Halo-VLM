# import torch
# #from torchcodec.decoders import VideoDecoder
# from transformers import AutoVideoProcessor, AutoModel
# import numpy as np

# processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
# model = AutoModel.from_pretrained(
#     "facebook/vjepa2-vitl-fpc64-256",
#     dtype=torch.float16,
#     device_map="auto",
#     attn_implementation="sdpa"
# )

# video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

# # vr = VideoDecoder(video_url)
# # frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
# # video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
# H, W = 224, 224  # example values
# # Random values from N(0, 1)
# x_randn = torch.randn(1, 3, H, W)

# video = processor(x_randn, return_tensors="pt").to(model.device)
# outputs = model(**video)

# # V-JEPA 2 encoder outputs, same as calling `model.get_vision_features()`
# encoder_outputs = outputs.last_hidden_state
# print(encoder_outputs.shape)

# # V-JEPA 2 predictor outputs
# predictor_outputs = outputs.predictor_output.last_hidden_state


import torch
from transformers import AutoVideoProcessor, AutoModel

class VJePA2Backbone(torch.nn.Module):
    def __init__(self, model_name="facebook/vjepa2-vitl-fpc64-256", H=224, W=224):
        super().__init__()
        self.H, self.W = H, W
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.eval()  # backbone typically in eval mode
        
    def forward(self, x):
        """
        x: torch.Tensor of shape [B, C, H, W] or [B, T, C, H, W]
           Single frame or video sequence
        Returns: model outputs (embeddings/features)
        """
        # Process input
        video_inputs = self.processor(x, return_tensors="pt").to(self.model.device)
        # Forward pass through VJePA2
        with torch.no_grad():  # inference mode for backbone
            outputs = self.model(**video_inputs)
        return outputs.last_hidden_state  # or outputs.pooler_output depending on use case
    