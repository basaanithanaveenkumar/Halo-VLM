import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import io
from datasets import load_dataset
import numpy as np


class EOVQADataset(Dataset):
    """
    DataLoader for EO-Data1.5M Visual Question Answering Dataset.
    
    Each original sample may contain multiple Q&A pairs (conversations).
    This dataset expands them so each Q&A pair becomes a separate sample.
    Images are mapped correctly based on <image> token usage across all Q&A pairs.
    
    Example:
        Original sample has 4 images and 2 Q&A pairs:
        - Q1: "What's in <image>?" (uses image 0)
        - Q2: "Compare <image> and <image>" (uses images 1, 2)
        
        This creates 2 samples:
        - Sample 1: images=[img0], question=Q1, answer=A1
        - Sample 2: images=[img1, img2], question=Q2, answer=A2
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform=None,
        max_images: int = 10,
        cache_dir: Optional[str] = None,
        streaming: bool = False
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            transform: Optional image transformation (e.g., from torchvision.transforms)
            max_images: Maximum number of images to handle per question
            cache_dir: Directory to cache the dataset
            streaming: Whether to stream the dataset (for large datasets)
        """
        self.transform = transform
        self.max_images = max_images
        self.split = split
        
        # Load the EO-Data1.5M dataset from HuggingFace
        print(f"Loading EO-Data1.5M dataset (split: {split})...")
        raw_dataset = load_dataset("IPEC-COMMUNITY/EO-Data1.5M",name="interleave-free_chat")
        raw_dataset= raw_dataset["train"]

        
        if streaming:
            print("Dataset loaded in streaming mode - cannot expand Q&A pairs in advance")
            self.samples = None
            self.dataset = raw_dataset
        else:
            print(f"Raw dataset loaded: {len(raw_dataset)} samples")
            print("Expanding Q&A pairs into individual samples...")
            self.samples = self._expand_qa_pairs(raw_dataset)
            print(f"Expanded to {len(self.samples)} Q&A pair samples")
            self.dataset = None
    
    def _expand_qa_pairs(self, raw_dataset) -> List[Dict]:
        """
        Expand each sample's Q&A pairs into separate samples.
        
        Each original sample may have multiple conversations (Q&A pairs).
        We track which images each Q&A pair uses based on <image> token positions.
        
        Dataset format:
        {
            'conversation': [
                {'from': 'human', 'value': '<image><image>Question 1?'},
                {'from': 'gpt', 'value': 'Answer 1'},
                {'from': 'human', 'value': '<image><image>Question 2?'},
                {'from': 'gpt', 'value': 'Answer 2'},
                ...
            ],
            'image': [img0, img1, img2, img3, ...],
            ...
        }
        
        Returns:
            List of expanded samples, each containing:
                - images: List of image objects for this Q&A pair
                - question: Question text
                - answer: Answer text
        """
        expanded_samples = []
        
        for raw_idx, raw_sample in enumerate(raw_dataset):
            # Get all images from this sample
            if 'image' in raw_sample:
                all_images = raw_sample['image'] if isinstance(raw_sample['image'], list) else [raw_sample['image']]
            elif 'images' in raw_sample:
                all_images = raw_sample['images']
            else:
                all_images = []
            
            # Get conversations (Q&A pairs)
            # Try different possible keys
            if 'conversation' in raw_sample:  # Note: singular 'conversation'
                conversations = raw_sample['conversation']
            elif 'conversations' in raw_sample:  # Note: plural 'conversations'
                conversations = raw_sample['conversations']
            else:
                # Fallback: try to extract single Q&A pair
                question = raw_sample.get('question', '')
                answer = raw_sample.get('answer', '')
                if question or answer:
                    conversations = [
                        {'from': 'human', 'value': question},
                        {'from': 'gpt', 'value': answer}
                    ]
                else:
                    conversations = []
            
            # Process conversations in pairs (question, answer)
            image_cursor = 0  # Tracks which image to use next
            
            for i in range(0, len(conversations), 2):
                if i + 1 >= len(conversations):
                    break  # Need both question and answer
                
                question_turn = conversations[i]
                answer_turn = conversations[i + 1]
                
                # Verify this is a human-gpt pair
                if question_turn.get('from') not in ['human', 'user']:
                    print(f"Warning: Sample {raw_idx}, turn {i}: Expected 'human'/'user', got '{question_turn.get('from')}'")
                    continue
                
                if answer_turn.get('from') not in ['gpt', 'assistant']:
                    print(f"Warning: Sample {raw_idx}, turn {i+1}: Expected 'gpt'/'assistant', got '{answer_turn.get('from')}'")
                    continue
                
                # Extract question and answer text
                question = question_turn.get('value', '')
                answer = answer_turn.get('value', '')
                
                # Count how many images this Q&A pair needs
                num_images_needed = self._count_image_tokens(question)
                
                # Get the images for this Q&A pair
                qa_images = []
                for j in range(num_images_needed):
                    if image_cursor + j < len(all_images):
                        qa_images.append(all_images[image_cursor + j])
                    else:
                        print(f"Warning: Sample {raw_idx}, Q&A {i//2}: Not enough images. "
                              f"Needed {num_images_needed}, available from cursor: {len(all_images) - image_cursor}")
                        break
                
                # Move cursor forward
                image_cursor += num_images_needed
                
                # Only add if we have content
                if question.strip() and answer.strip():
                    if len(answer)<24:
                        continue
                    else:
                        expanded_samples.append({
                            'images': qa_images,
                            'question': question,
                            'answer': answer,
                            'original_idx': raw_idx,
                            'qa_pair_idx': i // 2,
                            'source': raw_sample.get('source', '')
                        })
            if raw_idx>100:
                break
        
        return expanded_samples
    
    def _count_image_tokens(self, text: str) -> int:
        """Count number of <image> tokens in text"""
        return text.count('<image>')
    
    def _process_image(self, image: Union[Image.Image, bytes]) -> torch.Tensor:
        """
        Process a single image (PIL Image or bytes) into a tensor.
        
        Args:
            image: PIL Image or bytes
            
        Returns:
            torch.Tensor: Processed image tensor [C, H, W]
        """
        # Handle bytes input
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        return image_tensor
    
    def _stack_images(self, images: List[torch.Tensor], num_needed: int) -> torch.Tensor:
        """
        Stack images into a single tensor, handling variable numbers of images.
        
        Args:
            images: List of image tensors [C, H, W]
            num_needed: Number of images needed based on <image> tokens
            
        Returns:
            torch.Tensor: Stacked images [num_needed, C, H, W]
        """
        if len(images) < num_needed:
            # Pad with zeros if not enough images
            print(f"Warning: Expected {num_needed} images but got {len(images)}. Padding with zeros.")
            C, H, W = images[0].shape if images else (3, 224, 224)
            while len(images) < num_needed:
                images.append(torch.zeros(C, H, W))
        
        # Take only the required number of images in order
        images = images[:num_needed]
        
        # Stack along new dimension
        stacked = torch.stack(images, dim=0)  # [num_images, C, H, W]
        
        return stacked
    
    def __len__(self) -> int:
        """Return the number of Q&A pair samples in the dataset"""
        if self.samples is not None:
            return len(self.samples)
        else:
            raise TypeError("Streaming datasets don't support len(). Use dataset iteration instead.")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Get a single Q&A pair sample.
        
        Args:
            idx: Index of the Q&A pair sample
            
        Returns:
            image_tensor: Stacked images for this Q&A pair [num_images, C, H, W]
            question: Question text with <image> tokens
            answer: Answer text
        """
        if self.samples is None:
            raise RuntimeError("Cannot index streaming dataset. Use iteration instead.")
        
        # Get the expanded sample
        sample = self.samples[idx]
        
        question = sample['question']
        answer = sample['answer']
        images_raw = sample['images']
        
        # Count how many images are needed based on <image> tokens
        num_image_tokens = self._count_image_tokens(question)
        
        # Process all images for this Q&A pair
        processed_images = []
        for img in images_raw:
            try:
                processed_img = self._process_image(img)
                processed_images.append(processed_img)
            except Exception as e:
                print(f"Error processing image at sample {idx}: {e}")
                # Add a placeholder zero tensor
                C, H, W = (3, 224, 224)  # Default dimensions
                processed_images.append(torch.zeros(C, H, W))
        
        # Stack images according to the number needed
        if num_image_tokens > 0 and processed_images:
            image_tensor = self._stack_images(processed_images, num_image_tokens)
        else:
            # No images needed, return empty tensor
            C, H, W = (3, 224, 224) if not processed_images else processed_images[0].shape
            image_tensor = torch.zeros(0, C, H, W)
        
        return image_tensor, question, answer


def collate_fn(batch: List[Tuple[torch.Tensor, str, str]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function to handle variable number of images per sample.
    
    Args:
        batch: List of (image_tensor, question, answer) tuples
        
    Returns:
        Dictionary containing:
            - 'images': List of image tensors (variable num_images per sample)
            - 'questions': List of question strings
            - 'answers': List of answer strings
            - 'num_images': Tensor with number of images per sample
    """
    images = []
    questions = []
    answers = []
    num_images = []
    
    for img_tensor, question, answer in batch:
        images.append(img_tensor)
        questions.append(question)
        answers.append(answer)
        num_images.append(img_tensor.shape[0])
    
    return {
        'images': images,  # List of tensors with shape [num_images_i, C, H, W]
        'questions': questions,
        'answers': answers,
        'num_images': torch.tensor(num_images)
    }


def create_dataloader(
    split: str = 'train',
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    max_images: int = 10,
    cache_dir: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the EO-Data1.5M dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional image transformation
        max_images: Maximum number of images to handle per question
        cache_dir: Directory to cache the dataset
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = EOVQADataset(
        split=split,
        transform=transform,
        max_images=max_images,
        cache_dir=cache_dir,
        streaming=False
    )
    
    #dataset =  load_dataset("IPEC-COMMUNITY/EO-Data1.5M",name="interleave-free_chat")

    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Creating dataset...")
    dataset = load_dataset("IPEC-COMMUNITY/EO-Data1.5M",name="interleave-free_chat")
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = create_dataloader(
        split='train',
        batch_size=4,
        shuffle=True,
        num_workers=2,
        transform=transform
    )
    
    # Test batch
    print("\nTesting batch:")
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Number of samples in batch: {len(batch['questions'])}")
    print(f"Images per sample: {batch['num_images']}")
    for i, (imgs, q, a) in enumerate(zip(batch['images'], batch['questions'], batch['answers'])):
        print(f"\nSample {i}:")
        print(f"  Images shape: {imgs.shape}")
        print(f"  Question: {q[:100]}...")
        print(f"  Answer: {a[:100]}...")