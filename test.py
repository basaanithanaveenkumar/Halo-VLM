import torch
from torch.utils.data import IterableDataset, DataLoader
import io
from datasets import load_dataset
import cv2 # or use torchvision.io

class FineVideoAutomotive(IterableDataset):
    def __init__(self, target_category="Automotive"):
        self.ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
        self.target = target_category

    def __iter__(self):
        for row in self.ds:
            meta = json.loads(row['json'])
            if self.target.lower() in meta.get('content_parent_category', '').lower():
                # Extract video data
                video_bytes = row['mp4']
                
                # Creative Part: You can extract specific "Key Moments" 
                # FineVideo metadata includes timecoded scenes!
                scenes = meta.get('storyboard', [])
                import pdb;pdb.set_trace()
                
                yield {
                    "video": video_bytes,
                    "title": meta['title'],
                    "scenes": scenes,
                    "description": meta['description']
                }

# Initialize loader
creative_loader = DataLoader(FineVideoAutomotive(), batch_size=1)

import json
import io
import time
from datasets import load_dataset

def test_automotive_dataloader(num_samples=3):
    print(f"--- Starting FineVideo Automotive Test (Target: {num_samples} samples) ---")
    
    # 1. Initialize the streaming dataset
    try:
        ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
        print("Successfully connected to HuggingFace dataset server.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    found_count = 0
    start_time = time.time()

    # 2. Iterate and Filter
    for i, row in enumerate(ds):
        metadata = row['json']
        parent_cat = metadata.get('content_parent_category', 'N/A')
        fine_cat = metadata.get('content_fine_category', 'N/A')

        # Criteria: Look for Automotive or Vehicles keywords
        if "Automotive" in parent_cat or "Vehicles" in parent_cat:
            found_count += 1
            
            print(f"\n[Sample #{found_count}] (Index in stream: {i})")
            print(f"Title:       {metadata.get('title')}")
            print(f"Category:    {parent_cat} -> {fine_cat}")
            print(f"Video Size:  {len(row['mp4']) / 1024:.2f} KB")
            
            # Print a snippet of the storyboard (Creative element)
            storyboard = metadata.get('storyboard', [])
            if storyboard:
                print(f"Key Scene:   {storyboard[0].get('label', 'No label')}")
            
            # Verify video integrity
            video_stream = io.BytesIO(row['mp4'])
            if video_stream.getbuffer().nbytes > 0:
                print("Status:      Video Data Loaded Successfully ✅")
            else:
                print("Status:      Video Data Corrupt ❌")

        if found_count >= num_samples:
            break

    total_time = time.time() - start_time
    print(f"\n--- Test Complete in {total_time:.2f}s ---")

if __name__ == "__main__":
    test_automotive_dataloader()