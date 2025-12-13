from transformers import AutoTokenizer

from lavis.datasets.builders import load_dataset

import os
os.environ['cache_root'] = "/home/ha/.cache/lavis/coco"
# use the lavis coco dataset and create a dataloader suitable for vlm training
from dataloader import create_vlm_dataloaders    
coco_dataset = load_dataset("coco_caption", vis_path="/home/ha/.cache/lavis/coco")
        
print("\nCreating dataloaders with BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

dataloaders_with_tokenizer = create_vlm_dataloaders(
    coco_dataset,
    batch_size=16,
    num_workers=2,
    tokenizer=tokenizer,
    max_length=50
)

# Test with tokenizer
train_loader = dataloaders_with_tokenizer['train']
for batch in train_loader:
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Decoded text: {tokenizer.decode(batch['input_ids'][0])}")
    break