from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
import warnings
import json

# Load the dataset
print("Loading dataset...")
try:
    ds = load_dataset("Mutonix/Vript", "vript-short")
    train_dataset = ds['train']
    is_streaming = False
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load SmolLM tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")

# Check if pad_token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    warnings.warn(f"Pad token not found, using eos_token: {tokenizer.eos_token}")

class VriptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform=None, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        # Inspect the dataset structure
        self._inspect_dataset()
    
    def _inspect_dataset(self):
        """Inspect dataset structure"""
        print("\n=== Dataset Structure Inspection ===")
        try:
            if len(self.dataset) > 0:
                sample = self.dataset[0]
                import pdb;pdb.set_trace()
                print(f"Dataset length: {len(self.dataset)}")
                print(f"Sample keys: {list(sample.keys())}")
                
                for key, value in sample.items():
                    print(f"\n{key}:")
                    print(f"  Type: {type(value)}")
                    
                    if isinstance(value, str):
                        print(f"  Value preview: {value[:200]}")
                    elif isinstance(value, dict):
                        print(f"  Dict keys: {list(value.keys())}")
                        # Print first level of nested dict
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {type(subvalue)}")
                            if isinstance(subvalue, str):
                                print(f"      Preview: {subvalue[:100]}")
                    elif isinstance(value, list):
                        print(f"  List length: {len(value)}")
                        if len(value) > 0:
                            print(f"  First item type: {type(value[0])}")
            else:
                print("Dataset is empty")
        except Exception as e:
            print(f"Error inspecting dataset: {e}")
        print("=================================\n")
    
    def _extract_caption_text(self, caption_dict):
        """Extract text from the caption dictionary"""
        if not isinstance(caption_dict, dict):
            return str(caption_dict)
        
        # Based on the structure you showed, the caption dict has these keys:
        # 'shot_type', 'camera_movement', 'content', 'scene_title'
        
        parts = []
        
        # Add scene title if available
        if 'scene_title' in caption_dict and caption_dict['scene_title']:
            parts.append(f"Scene: {caption_dict['scene_title']}")
        
        # Add shot type if available
        if 'shot_type' in caption_dict and caption_dict['shot_type']:
            parts.append(f"Shot: {caption_dict['shot_type']}")
        
        # Add camera movement if available
        if 'camera_movement' in caption_dict and caption_dict['camera_movement']:
            parts.append(f"Camera: {caption_dict['camera_movement']}")
        
        # Add content (main description) - this is the most important part
        if 'content' in caption_dict and caption_dict['content']:
            parts.append(caption_dict['content'])
        
        # Join all parts with spaces
        if parts:
            return " ".join(parts)
        
        # If no parts found, try to extract any string values
        for value in caption_dict.values():
            if isinstance(value, str) and value.strip():
                return value
        
        return ""
    
    def _extract_video_path(self, metadata):
        """Extract video path from metadata"""
        # The dataset might have video paths in metadata
        # Check the meta field for video_id which might correspond to file paths
        if 'meta' in metadata and isinstance(metadata['meta'], dict):
            meta = metadata['meta']
            if 'video_id' in meta:
                # You might need to construct the path based on your setup
                video_id = meta['video_id']
                # Example: return f"videos/{video_id}.mp4"
                # For now, we'll return the ID
                return video_id
        
        # Check for clip_id
        if 'clip_id' in metadata and metadata['clip_id']:
            return metadata['clip_id']
        
        return None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract the actual caption text from the nested structure
        caption_text = ""
        
        # Method 1: Check if there's a 'caption' field that's a dict
        if 'caption' in item and isinstance(item['caption'], dict):
            caption_text = self._extract_caption_text(item['caption'])
        # Method 2: Check if there's a 'caption' field that's a string
        elif 'caption' in item and isinstance(item['caption'], str):
            caption_text = item['caption']
        # Method 3: Check for other text fields
        elif 'voiceover' in item and isinstance(item['voiceover'], str) and item['voiceover'].strip():
            caption_text = item['voiceover']
        # Method 4: Check metadata
        elif 'meta' in item and isinstance(item['meta'], dict):
            meta = item['meta']
            if 'video_title' in meta and isinstance(meta['video_title'], str):
                caption_text = meta['video_title']
        
        # If we still don't have a caption, use clip_id as fallback
        if not caption_text.strip() and 'clip_id' in item:
            caption_text = item['clip_id']
        
        # Clean up the text
        caption_text = caption_text.strip()
        
        # Debug for first few samples
        if idx < 3:
            print(f"\nSample {idx}:")
            print(f"  Raw item keys: {list(item.keys())}")
            if 'caption' in item:
                print(f"  Caption field type: {type(item['caption'])}")
                if isinstance(item['caption'], dict):
                    print(f"  Caption dict keys: {list(item['caption'].keys())}")
            print(f"  Extracted caption: {caption_text[:200]}...")
        
        # Tokenize the caption
        try:
            encoding = self.tokenizer(
                caption_text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True
            )
        except Exception as e:
            print(f"Error tokenizing caption: {e}")
            # Fallback to empty string
            encoding = self.tokenizer(
                "",
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True
            )
        
        # Handle video - Vript likely has video paths, not actual video tensors
        # You'll need to load the video files based on the metadata
        video_path = self._extract_video_path(item)
        video_tensor = None
        
        # If you have a transform function that loads videos from paths
        if self.transform and video_path is not None:
            try:
                # This transform should handle loading video from path
                video_tensor = self.transform(video_path)
            except Exception as e:
                warnings.warn(f"Video loading failed for {video_path}: {e}")
                video_tensor = None
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'caption': caption_text,
            'video': video_tensor,
            'video_path': video_path,  # Keep path for reference
            'metadata': item
        }

# Custom collate function
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, 
        batch_first=True, 
        padding_value=0
    )
    
    # Handle videos - only stack if all have videos
    videos = []
    video_paths = []
    
    for item in batch:
        if item['video'] is not None:
            videos.append(item['video'])
        video_paths.append(item['video_path'])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'captions': [item['caption'] for item in batch],
        'videos': torch.stack(videos) if len(videos) == len(batch) and len(videos) > 0 else None,
        'video_paths': video_paths,
        'metadata': [item['metadata'] for item in batch]
    }

# Create dataset
print("\nCreating VriptDataset...")
vript_pytorch = VriptDataset(
    train_dataset, 
    tokenizer,
    max_length=512
)

# Create DataLoader
print("\nCreating DataLoader...")
dataloader = DataLoader(
    vript_pytorch,
    batch_size=4,
    shuffle=True,
    num_workers=0,  # Set to 0 for debugging
    collate_fn=collate_fn,
    pin_memory=False
)

# Test the dataloader
print("\n=== Testing DataLoader ===")
try:
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        captions = batch['captions']
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Batch size: {len(captions)}")
        
        # Show decoded text
        for i in range(min(2, len(captions))):  # Show first 2 captions
            print(f"\n  Sample {i}:")
            print(f"    Caption length: {len(captions[i])} chars")
            print(f"    Caption preview: {captions[i][:200]}...")
            
            decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            print(f"    Decoded tokens: {decoded[:200]}...")
            
            # Show metadata info
            if 'meta' in batch['metadata'][i]:
                meta = batch['metadata'][i]['meta']
                print(f"    Video ID: {meta.get('video_id', 'N/A')}")
                print(f"    Video title: {meta.get('video_title', 'N/A')[:100]}...")
        
        # Check video paths
        print(f"\n  Video paths in batch: {batch['video_paths']}")
        
        if batch_idx >= 1:  # Test with first 2 batches
            print("\nDataLoader test successful!")
            break
            
except Exception as e:
    print(f"\nError in dataloader: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# If you need to actually load videos, here's a simple transform example:
print("\n" + "="*50)
print("VIDEO LOADING GUIDANCE")
print("="*50)
