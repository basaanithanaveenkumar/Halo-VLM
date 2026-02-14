"""
Training Visualization System

Generates video frames showing:
- Input images (multiple per sample)
- Predicted vs target embeddings (t-SNE projection)
- Similarity matrix heatmap
- Training metrics over time

Output: Video at 1 FPS showing training progress
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
from datetime import datetime
from collections import deque


class TrainingVisualizer:
    """
    Visualizes training progress with image inputs, embeddings, and similarity matrices.
    Generates frames that can be compiled into a video at 1 FPS.
    """
    
    def __init__(
        self,
        output_dir="./visualizations",
        fps=1,
        figsize=(20, 12),
        dpi=100,
        max_samples_to_show=8,
        max_images_per_sample=4,
        history_length=100
    ):
        """
        Args:
            output_dir: Directory to save frames and video
            fps: Frames per second for output video
            figsize: Figure size for visualization
            dpi: DPI for saved frames
            max_samples_to_show: Maximum samples to display from batch
            max_images_per_sample: Maximum images to show per sample
            history_length: Number of historical metrics to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self.max_samples = max_samples_to_show
        self.max_images = max_images_per_sample
        
        # History tracking
        self.loss_history = deque(maxlen=history_length)
        self.similarity_history = deque(maxlen=history_length)
        self.step_history = deque(maxlen=history_length)
        
        # Frame counter
        self.frame_count = 0
        
        # Color schemes
        self.cmap_similarity = 'RdYlGn'
        self.cmap_embeddings = 'tab20'
        
        print(f"Visualizer initialized. Frames will be saved to: {self.frames_dir}")
    
    def denormalize_image(self, img_tensor):
        """
        Denormalize image tensor for display.
        
        Args:
            img_tensor: [C, H, W] normalized tensor
            
        Returns:
            numpy array [H, W, C] in range [0, 255]
        """
        # ImageNet normalization stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert to numpy and transpose
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        img = img * std + mean
        
        # Clip and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img
    
    def compute_similarity_matrix(self, pred_embeddings, target_embeddings):
        """
        Compute cosine similarity matrix between predictions and targets.
        
        Args:
            pred_embeddings: [batch_size, embedding_dim]
            target_embeddings: [batch_size, embedding_dim]
            
        Returns:
            similarity_matrix: [batch_size, batch_size]
        """
        # Normalize
        pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        
        # Compute similarity
        similarity = (pred_norm @ target_norm.T).cpu().numpy()
        
        return similarity
    
    def project_embeddings_2d(self, pred_embeddings, target_embeddings, method='tsne'):
        """
        Project embeddings to 2D for visualization.
        
        Args:
            pred_embeddings: [batch_size, embedding_dim]
            target_embeddings: [batch_size, embedding_dim]
            method: 'tsne' or 'pca'
            
        Returns:
            pred_2d: [batch_size, 2]
            target_2d: [batch_size, 2]
        """
        # Combine embeddings
        all_embeddings = torch.cat([pred_embeddings, target_embeddings], dim=0).cpu().numpy()
        
        # Project to 2D
        if method == 'tsne':
            # Use PCA first to reduce dimensionality for faster t-SNE
            if all_embeddings.shape[1] > 50:
                pca = PCA(n_components=50)
                all_embeddings = pca.fit_transform(all_embeddings)
            
            projector = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
        else:  # pca
            projector = PCA(n_components=2)
        
        embeddings_2d = projector.fit_transform(all_embeddings)
        
        # Split back
        batch_size = len(pred_embeddings)
        pred_2d = embeddings_2d[:batch_size]
        target_2d = embeddings_2d[batch_size:]
        
        return pred_2d, target_2d
    
    def create_visualization_frame(
        self,
        images_batch,
        questions,
        answers,
        pred_embeddings,
        target_embeddings,
        loss,
        step,
        epoch
    ):
        """
        Create a single visualization frame.
        
        Args:
            images_batch: [batch_size, num_images, C, H, W]
            questions: List of question strings
            answers: List of answer strings
            pred_embeddings: [batch_size, embedding_dim]
            target_embeddings: [batch_size, embedding_dim]
            loss: Current loss value
            step: Current training step
            epoch: Current epoch
            
        Returns:
            frame: numpy array of the visualization
        """
        # Limit number of samples to display
        batch_size = min(len(questions), self.max_samples)
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
            f"Training Visualization - Epoch {epoch} | Step {step} | Loss: {loss:.4f}",
            fontsize=16, fontweight='bold'
        )
        
        # ====================================================================
        # 1. Input Images Grid (Top Left)
        # ====================================================================
        ax_images = fig.add_subplot(gs[0:2, 0])
        ax_images.set_title("Input Images", fontsize=12, fontweight='bold')
        ax_images.axis('off')
        
        self._plot_images_grid(ax_images, images_batch[:batch_size], questions[:batch_size])
        
        # ====================================================================
        # 2. Similarity Matrix (Top Middle)
        # ====================================================================
        ax_similarity = fig.add_subplot(gs[0:2, 1])
        ax_similarity.set_title("Similarity Matrix (Predicted × Target)", fontsize=12, fontweight='bold')
        
        similarity_matrix = self.compute_similarity_matrix(
            pred_embeddings[:batch_size],
            target_embeddings[:batch_size]
        )
        
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap=self.cmap_similarity,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax_similarity
        )
        ax_similarity.set_xlabel("Target Embedding Index")
        ax_similarity.set_ylabel("Predicted Embedding Index")
        
        # Highlight diagonal (correct pairs)
        for i in range(min(batch_size, similarity_matrix.shape[0])):
            ax_similarity.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))
        
        # ====================================================================
        # 3. Embedding Space Visualization (Top Right)
        # ====================================================================
        ax_embeddings = fig.add_subplot(gs[0:2, 2])
        ax_embeddings.set_title("Embedding Space (2D Projection)", fontsize=12, fontweight='bold')
        
        try:
            pred_2d, target_2d = self.project_embeddings_2d(
                pred_embeddings[:batch_size],
                target_embeddings[:batch_size],
                method='pca'  # Use PCA for speed
            )
            
            # Plot predicted embeddings
            ax_embeddings.scatter(
                pred_2d[:, 0], pred_2d[:, 1],
                c=range(batch_size), cmap=self.cmap_embeddings,
                marker='o', s=200, alpha=0.6, edgecolors='black', linewidth=2,
                label='Predicted'
            )
            
            # Plot target embeddings
            ax_embeddings.scatter(
                target_2d[:, 0], target_2d[:, 1],
                c=range(batch_size), cmap=self.cmap_embeddings,
                marker='s', s=200, alpha=0.6, edgecolors='black', linewidth=2,
                label='Target'
            )
            
            # Draw lines connecting corresponding pairs
            for i in range(batch_size):
                ax_embeddings.plot(
                    [pred_2d[i, 0], target_2d[i, 0]],
                    [pred_2d[i, 1], target_2d[i, 1]],
                    'k--', alpha=0.3, linewidth=1
                )
                
                # Annotate with sample index
                ax_embeddings.annotate(
                    str(i),
                    xy=(pred_2d[i, 0], pred_2d[i, 1]),
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='center'
                )
            
            ax_embeddings.legend(loc='best')
            ax_embeddings.grid(True, alpha=0.3)
            ax_embeddings.set_xlabel("Component 1")
            ax_embeddings.set_ylabel("Component 2")
            
        except Exception as e:
            ax_embeddings.text(0.5, 0.5, f"Projection failed: {str(e)}", 
                             ha='center', va='center', transform=ax_embeddings.transAxes)
        
        # ====================================================================
        # 4. Question-Answer Pairs (Bottom Left)
        # ====================================================================
        ax_qa = fig.add_subplot(gs[2:, 0])
        ax_qa.set_title("Question-Answer Pairs", fontsize=12, fontweight='bold')
        ax_qa.axis('off')
        
        qa_text = ""
        for i in range(batch_size):
            qa_text += f"Sample {i}:\n"
            qa_text += f"Q: {questions[i][:80]}{'...' if len(questions[i]) > 80 else ''}\n"
            qa_text += f"A: {answers[i][:80]}{'...' if len(answers[i]) > 80 else ''}\n\n"
        
        ax_qa.text(0.05, 0.95, qa_text, transform=ax_qa.transAxes,
                  fontsize=8, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # ====================================================================
        # 5. Loss History (Bottom Middle)
        # ====================================================================
        ax_loss = fig.add_subplot(gs[2:, 1])
        ax_loss.set_title("Loss History", fontsize=12, fontweight='bold')
        
        # Update history
        self.loss_history.append(loss)
        self.step_history.append(step)
        
        if len(self.loss_history) > 1:
            ax_loss.plot(list(self.step_history), list(self.loss_history), 
                        'b-', linewidth=2, label='Loss')
            ax_loss.scatter(step, loss, c='red', s=100, zorder=5, label='Current')
            ax_loss.set_xlabel("Step")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
        else:
            ax_loss.text(0.5, 0.5, "Collecting data...", 
                        ha='center', va='center', transform=ax_loss.transAxes)
        
        # ====================================================================
        # 6. Similarity Statistics (Bottom Right)
        # ====================================================================
        ax_stats = fig.add_subplot(gs[2:, 2])
        ax_stats.set_title("Similarity Statistics", fontsize=12, fontweight='bold')
        ax_stats.axis('off')
        
        # Compute statistics
        diagonal_sim = np.diagonal(similarity_matrix).mean()
        off_diagonal_sim = (similarity_matrix.sum() - np.diagonal(similarity_matrix).sum()) / (
            similarity_matrix.size - len(similarity_matrix)
        )
        
        # Update history
        self.similarity_history.append(diagonal_sim)
        
        stats_text = f"""
Current Batch Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━
Diagonal Similarity (Correct Pairs):
  Mean: {diagonal_sim:.4f}
  
Off-Diagonal Similarity (Wrong Pairs):
  Mean: {off_diagonal_sim:.4f}
  
Margin (Diagonal - Off-Diagonal):
  {diagonal_sim - off_diagonal_sim:.4f}
  
Average Over Last {len(self.similarity_history)} Steps:
  Diagonal: {np.mean(self.similarity_history):.4f}
  
Individual Sample Similarities:
"""
        for i in range(min(batch_size, len(similarity_matrix))):
            stats_text += f"  Sample {i}: {similarity_matrix[i, i]:.4f}\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # ====================================================================
        # Convert figure to image
        # ====================================================================
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        frame = np.array(Image.open(buf))
        buf.close()
        plt.close(fig)
        
        return frame
    
    def _plot_images_grid(self, ax, images_batch, questions):
        """
        Plot a grid of images with questions.
        
        Args:
            ax: Matplotlib axis
            images_batch: [batch_size, num_images, C, H, W]
            questions: List of question strings
        """
        batch_size = len(images_batch)
        
        # Create a grid layout
        rows = batch_size
        cols = self.max_images
        
        # Create combined image
        img_height, img_width = 100, 100  # Target size for each image
        grid_height = rows * img_height
        grid_width = cols * img_width
        
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        for i in range(batch_size):
            num_images = min(images_batch[i].shape[0], self.max_images)
            
            for j in range(num_images):
                # Get image
                img = self.denormalize_image(images_batch[i][j])
                
                # Resize to fixed size
                img_resized = cv2.resize(img, (img_width, img_height))
                
                # Place in grid
                y_start = i * img_height
                x_start = j * img_width
                grid[y_start:y_start+img_height, x_start:x_start+img_width] = img_resized
                
                # Add border
                cv2.rectangle(grid, 
                            (x_start, y_start), 
                            (x_start+img_width, y_start+img_height),
                            (0, 0, 0), 2)
                
                # Add sample index on first image
                if j == 0:
                    cv2.putText(grid, f"S{i}", 
                              (x_start + 5, y_start + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        ax.imshow(grid)
        
        # Add question labels on the right
        for i in range(batch_size):
            question_short = questions[i][:40] + '...' if len(questions[i]) > 40 else questions[i]
            num_tokens = questions[i].count('<image>')
            
            y_pos = (i + 0.5) / batch_size
            ax.text(1.02, y_pos, f"Q{i}: {question_short}\n({num_tokens} images)",
                   transform=ax.transAxes, fontsize=8, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    def save_frame(self, frame, step):
        """
        Save a frame to disk.
        
        Args:
            frame: numpy array of the frame
            step: Current step number
        """
        frame_path = self.frames_dir / f"frame_{step:06d}.png"
        Image.fromarray(frame).save(frame_path)
        self.frame_count += 1
        
        return frame_path
    
    def create_video(self, output_name="training_visualization.mp4", cleanup_frames=False):
        """
        Compile all frames into a video.
        
        Args:
            output_name: Name of output video file
            cleanup_frames: Whether to delete frames after creating video
            
        Returns:
            Path to created video
        """
        frame_files = sorted(self.frames_dir.glob("frame_*.png"))
        
        if len(frame_files) == 0:
            print("No frames found to create video!")
            return None
        
        print(f"Creating video from {len(frame_files)} frames...")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Create video writer
        video_path = self.output_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        
        # Write frames
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            video_writer.write(frame)
        
        video_writer.release()
        
        print(f"Video created: {video_path}")
        print(f"Duration: {len(frame_files) / self.fps:.1f} seconds at {self.fps} FPS")
        
        # Cleanup frames if requested
        if cleanup_frames:
            print("Cleaning up frames...")
            for frame_file in frame_files:
                frame_file.unlink()
            print("Frames deleted.")
        
        return video_path


# ============================================================================
# Integration with Training Loop
# ============================================================================

def visualize_batch(
    visualizer,
    images,
    questions,
    answers,
    pred_embeddings,
    target_embeddings,
    loss,
    step,
    epoch,
    save_frame=True
):
    """
    Visualize a single batch during training.
    
    Args:
        visualizer: TrainingVisualizer instance
        images: [batch_size, num_images, C, H, W]
        questions: List of question strings
        answers: List of answer strings
        pred_embeddings: [batch_size, embedding_dim]
        target_embeddings: [batch_size, embedding_dim]
        loss: Current loss value
        step: Current step
        epoch: Current epoch
        save_frame: Whether to save the frame
        
    Returns:
        frame: numpy array of visualization
    """
    frame = visualizer.create_visualization_frame(
        images_batch=images,
        questions=questions,
        answers=answers,
        pred_embeddings=pred_embeddings,
        target_embeddings=target_embeddings,
        loss=loss,
        step=step,
        epoch=epoch
    )
    
    if save_frame:
        visualizer.save_frame(frame, step)
    
    return frame


if __name__ == "__main__":
    print("Training Visualizer Module")
    print("="*80)
    print("This module provides visualization capabilities for VQA training.")
    print("\nUsage:")
    print("  from training_visualizer import TrainingVisualizer, visualize_batch")
    print("  visualizer = TrainingVisualizer()")
    print("  # In training loop:")
    print("  visualize_batch(visualizer, images, questions, answers, ...)")
    print("  # After training:")
    print("  visualizer.create_video()")