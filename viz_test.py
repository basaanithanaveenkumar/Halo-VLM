import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

def process_image_tensors(image_tensors, normalize=True):
    """
    Process image tensors for visualization
    
    Args:
        image_tensors: torch.Tensor of shape (batch_size, 3, height, width)
        normalize: Whether to denormalize the images
    Returns:
        numpy array of images ready for display
    """
    batch_size = image_tensors.shape[0]
    
    # Move to CPU if on GPU
    if image_tensors.is_cuda:
        image_tensors = image_tensors.cpu()
    
    # Convert to numpy and change from (C, H, W) to (H, W, C)
    images_np = image_tensors.numpy()
    
    # Transpose from (B, C, H, W) to (B, H, W, C)
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # Denormalize if needed (assuming normalization to [0, 1] or [-1, 1])
    if normalize:
        # Check if values are in [0, 1] or [-1, 1]
        if images_np.min() < 0:
            # From [-1, 1] to [0, 1]
            images_np = (images_np + 1) / 2
        # Clip to valid range
        images_np = np.clip(images_np, 0, 1)
    
    return images_np

def visualize_embeddings_with_images(target_embeddings, predicted_embeddings, 
                                    image_tensors, answer_texts=None,
                                    use_pca=True, perplexity=30, random_state=42,
                                    save_path=None, dpi=300, figsize=(16, 12)):
    """
    Visualize embeddings with corresponding images as thumbnails
    
    Args:
        target_embeddings: numpy array of shape (batch_size, embedding_dim)
        predicted_embeddings: numpy array of shape (batch_size, embedding_dim)
        image_tensors: torch.Tensor of shape (batch_size, 3, height, width)
        answer_texts: list of strings or None
        use_pca: if True, use PCA; if False, use t-SNE
        perplexity: parameter for t-SNE (if used)
        random_state: random seed for reproducibility
        save_path: Path to save the figure
        dpi: Resolution for saved image
        figsize: Figure size (width, height) in inches
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Answer {i}" for i in range(batch_size)]
    
    # Combine target and predicted embeddings for consistent scaling
    all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        # Use PCA for dimensionality reduction
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(reducer.explained_variance_ratio_):.2%}")
    else:
        # Use t-SNE for dimensionality reduction
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                      random_state=random_state, n_iter=1000)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Split back into target and predicted
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Process image tensors
    images_np = process_image_tensors(image_tensors)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot target embeddings as solid circles
    target_scatter = ax.scatter(target_2d[:, 0], target_2d[:, 1], 
                                c='blue', marker='o', s=200, alpha=0.7, 
                                label='Target Embeddings', edgecolors='black', linewidth=2)
    
    # Plot predicted embeddings as squares
    predicted_scatter = ax.scatter(predicted_2d[:, 0], predicted_2d[:, 1], 
                                   c='red', marker='s', s=200, alpha=0.7, 
                                   label='Predicted Embeddings', edgecolors='black', linewidth=2)
    
    # Draw lines connecting target to predicted
    lines = []
    for i in range(batch_size):
        line = ax.plot([target_2d[i, 0], predicted_2d[i, 0]], 
                       [target_2d[i, 1], predicted_2d[i, 1]], 
                       'k--', alpha=0.4, linewidth=1.5)[0]
        lines.append(line)
    
    # Create annotations with thumbnails
    annotations = []
    thumbnail_size = 0.15  # Size of thumbnails relative to axis
    
    for i in range(batch_size):
        # Add thumbnail near predicted embedding
        img_extent = [
            predicted_2d[i, 0] - thumbnail_size,
            predicted_2d[i, 0] + thumbnail_size,
            predicted_2d[i, 1] - thumbnail_size,
            predicted_2d[i, 1] + thumbnail_size
        ]
        
        ax.imshow(images_np[i], aspect='auto', extent=img_extent, zorder=5)
        
        # Add text annotation
        if len(answer_texts[i]) > 30:
            text = answer_texts[i][:27] + "..."
        else:
            text = answer_texts[i]
            
        anno = ax.annotate(text, 
                          xy=(predicted_2d[i, 0], predicted_2d[i, 1] + thumbnail_size),
                          xytext=(0, 10), textcoords='offset points',
                          fontsize=9, ha='center',
                          bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="yellow", alpha=0.8))
        annotations.append(anno)
    
    # Add index numbers to target embeddings
    for i in range(batch_size):
        ax.annotate(f'T{i}', 
                   xy=(target_2d[i, 0], target_2d[i, 1]),
                   xytext=(0, -25), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue',
                   ha='center')
    
    method = "PCA" if use_pca else f"t-SNE (perplexity={perplexity})"
    ax.set_title(f'2D Visualization of Embeddings with Images ({method})', fontsize=16)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return target_2d, predicted_2d, fig

def create_embedding_video(target_embeddings, predicted_embeddings, 
                          image_tensors, answer_texts=None,
                          use_pca=True, random_state=42,
                          save_path="./embedding_video.gif", 
                          fps=2, figsize=(12, 10)):
    """
    Create an animated video showing embeddings and images sequentially
    
    Args:
        target_embeddings: numpy array of shape (batch_size, embedding_dim)
        predicted_embeddings: numpy array of shape (batch_size, embedding_dim)
        image_tensors: torch.Tensor of shape (batch_size, 3, height, width)
        answer_texts: list of strings or None
        use_pca: if True, use PCA; if False, use t-SNE
        save_path: Path to save the animated GIF
        fps: Frames per second for the animation
        figsize: Figure size (width, height) in inches
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Answer {i}" for i in range(batch_size)]
    
    # Process image tensors
    images_np = process_image_tensors(image_tensors)
    
    # Reduce dimensionality
    all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set up embedding plot
    target_scatter = ax1.scatter([], [], c='blue', marker='o', s=150, 
                                 label='Target', alpha=0.7, edgecolors='black')
    predicted_scatter = ax1.scatter([], [], c='red', marker='s', s=150, 
                                    label='Predicted', alpha=0.7, edgecolors='black')
    
    # Current sample indicator
    current_target = ax1.scatter([], [], c='green', marker='*', s=300, 
                                 label='Current Target', alpha=1.0, edgecolors='black')
    current_predicted = ax1.scatter([], [], c='orange', marker='*', s=300, 
                                    label='Current Predicted', alpha=1.0, edgecolors='black')
    
    # Connection line for current sample
    connection_line = ax1.plot([], [], 'g-', linewidth=2, alpha=0.8)[0]
    
    # Set limits with padding
    all_points = np.vstack([target_2d, predicted_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_title(f'Embedding Space ({method})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Set up image display
    ax2.axis('off')
    image_display = ax2.imshow(np.ones((224, 224, 3)))
    ax2.set_title('Current Image')
    
    # Text display
    text_display = ax2.text(0.5, -0.1, '', transform=ax2.transAxes, 
                           ha='center', va='top', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Info text
    info_text = fig.text(0.5, 0.01, '', ha='center', fontsize=11)
    
    # Update function for animation
    def update(frame):
        idx = frame % batch_size
        
        # Update embedding plot
        target_scatter.set_offsets(target_2d[:idx+1])
        predicted_scatter.set_offsets(predicted_2d[:idx+1])
        
        # Highlight current sample
        current_target.set_offsets([target_2d[idx]])
        current_predicted.set_offsets([predicted_2d[idx]])
        
        # Update connection line
        connection_line.set_data([target_2d[idx, 0], predicted_2d[idx, 0]], 
                                 [target_2d[idx, 1], predicted_2d[idx, 1]])
        
        # Update image
        ax2.clear()
        ax2.imshow(images_np[idx])
        ax2.axis('off')
        ax2.set_title(f'Image {idx}')
        
        # Update text
        if len(answer_texts[idx]) > 50:
            display_text = answer_texts[idx][:47] + "..."
        else:
            display_text = answer_texts[idx]
        text_display.set_text(display_text)
        
        # Update info text
        distance = np.linalg.norm(target_embeddings[idx] - predicted_embeddings[idx])
        cos_sim = np.dot(target_embeddings[idx], predicted_embeddings[idx]) / (
            np.linalg.norm(target_embeddings[idx]) * np.linalg.norm(predicted_embeddings[idx])
        )
        info_text.set_text(f'Sample {idx}: Distance = {distance:.3f}, Cosine Similarity = {cos_sim:.3f}')
        
        return [target_scatter, predicted_scatter, current_target, 
                current_predicted, connection_line, image_display, 
                text_display, info_text]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=batch_size, 
                                 interval=1000//fps, blit=True)
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # For GIF
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps, dpi=100)
        # For MP4 (requires ffmpeg)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            save_path = save_path + '.gif'
            ani.save(save_path, writer='pillow', fps=fps, dpi=100)
        
        print(f"Animation saved to: {save_path}")
    
    return ani

def visualize_embeddings_interactive_with_images(target_embeddings, predicted_embeddings,
                                                image_tensors, answer_texts=None,
                                                use_pca=True, save_path=None,
                                                width=1200, height=800):
    """
    Create interactive visualization with image hover functionality
    
    Args:
        save_path: Path to save the interactive HTML file
        width: Width of the plot in pixels
        height: Height of the plot in pixels
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Answer {i}" for i in range(batch_size)]
    
    # Process image tensors
    images_np = process_image_tensors(image_tensors)
    
    # Convert images to base64 for HTML embedding
    import base64
    from io import BytesIO
    
    image_base64 = []
    for i in range(batch_size):
        img = Image.fromarray((images_np[i] * 255).astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_base64.append(f"data:image/png;base64,{img_str}")
    
    # Reduce dimensionality
    all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=(f'Embedding Space ({method})', 'Image Preview'),
        horizontal_spacing=0.05
    )
    
    # Add target embeddings
    fig.add_trace(go.Scatter(
        x=target_2d[:, 0],
        y=target_2d[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='blue', symbol='circle'),
        text=[f'T{i}' for i in range(batch_size)],
        textposition="top center",
        name='Target Embeddings',
        hoverinfo='text',
        hovertext=[f'Target {i}' for i in range(batch_size)],
        customdata=[image_base64[i] for i in range(batch_size)]
    ), row=1, col=1)
    
    # Add predicted embeddings
    fig.add_trace(go.Scatter(
        x=predicted_2d[:, 0],
        y=predicted_2d[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='square'),
        text=[f'P{i}' for i in range(batch_size)],
        textposition="bottom center",
        name='Predicted Embeddings',
        hoverinfo='text',
        hovertext=[f'Predicted {i}: {txt[:50]}{"..." if len(txt) > 50 else ""}' 
                  for i, txt in enumerate(answer_texts)],
        customdata=[image_base64[i] for i in range(batch_size)]
    ), row=1, col=1)
    
    # Add connecting lines
    for i in range(batch_size):
        fig.add_trace(go.Scatter(
            x=[target_2d[i, 0], predicted_2d[i, 0]],
            y=[target_2d[i, 1], predicted_2d[i, 1]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='none'
        ), row=1, col=1)
    
    # Add initial image to preview pane
    fig.add_trace(go.Image(
        z=images_np[0],
        name='Image Preview',
        hoverinfo='none'
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Embedding Visualization with Images',
        width=width,
        height=height,
        hovermode='closest',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Component 1", row=1, col=1)
    fig.update_yaxes(title_text="Component 2", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    
    # Add JavaScript for image update on hover
    fig.update_traces(
        selector=dict(name='Target Embeddings'),
        hovertemplate="<b>Target %{text}</b><br>" +
                     "Component 1: %{x:.3f}<br>" +
                     "Component 2: %{y:.3f}<extra></extra>"
    )
    
    fig.update_traces(
        selector=dict(name='Predicted Embeddings'),
        hovertemplate="<b>Predicted %{text}</b><br>" +
                     "%{hovertext}<br>" +
                     "Component 1: %{x:.3f}<br>" +
                     "Component 2: %{y:.3f}<extra></extra>"
    )
    
    # Save interactive HTML if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if not save_path.endswith('.html'):
            save_path = save_path + '.html'
        
        fig.write_html(save_path, include_plotlyjs=True)
        print(f"Interactive plot saved to: {save_path}")
    
    return fig

# def create_sample_image_data():
#     """Create sample embeddings and image tensors"""
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     batch_size = 6
#     embedding_dim = 512
    
#     # Create sample embeddings
#     target_embeddings = np.random.randn(batch_size, embedding_dim) * 0.5
#     predicted_embeddings = target_embeddings + np.random.randn(batch_size, embedding_dim) * 0.3
    
#     # Create sample image tensors (batch_size, 3, 244, 244)
#     image_tensors = torch.randn(batch_size, 3, 224, 224) * 0.5 + 0.5
#     image_tensors = torch.clamp(image_tensors, 0, 1)
    
#     # Sample answer texts
#     answer_texts = [
#         "A beautiful sunset over mountains",
#         "A cat sitting on a windowsill",
#         "A bowl of fresh fruit",
#         "A city skyline at night",
#         "A forest path in autumn",
#         "Ocean waves crashing on shore"
#     ]
    
#     return target_embeddings, predicted_embeddings, image_tensors, answer_texts
def process_video_tensors(video_tensors_list, normalize=True):
    """
    Process video tensors for visualization
    
    Args:
        video_tensors_list: list of torch.Tensor, each of shape (num_frames, 3, height, width)
        normalize: Whether to denormalize the images
    Returns:
        list of numpy arrays of videos ready for display
    """
    processed_videos = []
    
    for video_tensor in video_tensors_list:
        # Move to CPU if on GPU
        if video_tensor.is_cuda:
            video_tensor = video_tensor.cpu()
        
        # Convert to numpy and change from (F, C, H, W) to (F, H, W, C)
        video_np = video_tensor.numpy()
        
        # Transpose from (F, C, H, W) to (F, H, W, C) if needed
        if len(video_np.shape) == 4 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        
        # Denormalize if needed
        if normalize:
            if video_np.min() < 0:
                video_np = (video_np + 1) / 2
            video_np = np.clip(video_np, 0, 1)
        
        processed_videos.append(video_np)
    
    return processed_videos
def visualize_embeddings_with_videos(target_embeddings, predicted_embeddings, 
                                    video_tensors_list, answer_texts=None,
                                    use_pca=True, perplexity=30, random_state=42,
                                    save_path=None, dpi=300, figsize=(16, 12),
                                    show_keyframe=True, keyframe_index=0):
    """
    Visualize embeddings with corresponding videos (show keyframe)
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Video {i}" for i in range(batch_size)]
    
    # Combine target and predicted embeddings for consistent scaling
    all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                      random_state=random_state, n_iter=1000)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Process video tensors
    videos_np = process_video_tensors(video_tensors_list)
    
    # Get keyframes for thumbnails
    keyframes = []
    for video_np in videos_np:
        if keyframe_index == -1:
            frame_idx = len(video_np) - 1
        else:
            frame_idx = min(keyframe_index, len(video_np) - 1)
        keyframes.append(video_np[frame_idx])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings (same as before)
    target_scatter = ax.scatter(target_2d[:, 0], target_2d[:, 1], 
                                c='blue', marker='o', s=200, alpha=0.7, 
                                label='Target Embeddings', edgecolors='black', linewidth=2)
    
    predicted_scatter = ax.scatter(predicted_2d[:, 0], predicted_2d[:, 1], 
                                   c='red', marker='s', s=200, alpha=0.7, 
                                   label='Predicted Embeddings', edgecolors='black', linewidth=2)
    
    # Draw connecting lines
    for i in range(batch_size):
        ax.plot([target_2d[i, 0], predicted_2d[i, 0]], 
                [target_2d[i, 1], predicted_2d[i, 1]], 
                'k--', alpha=0.4, linewidth=1.5)
    
    # Add keyframe thumbnails with video info
    thumbnail_size = 0.15
    
    for i in range(batch_size):
        # Add keyframe thumbnail
        img_extent = [
            predicted_2d[i, 0] - thumbnail_size,
            predicted_2d[i, 0] + thumbnail_size,
            predicted_2d[i, 1] - thumbnail_size,
            predicted_2d[i, 1] + thumbnail_size
        ]
        
        ax.imshow(keyframes[i], aspect='auto', extent=img_extent, zorder=5)
        
        # Add video info annotation
        video_info = f"{answer_texts[i]}\nFrames: {len(videos_np[i])}"
        if len(video_info) > 40:
            text = video_info[:37] + "..."
        else:
            text = video_info
            
        ax.annotate(text, 
                   xy=(predicted_2d[i, 0], predicted_2d[i, 1] + thumbnail_size),
                   xytext=(0, 10), textcoords='offset points',
                   fontsize=8, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor="yellow", alpha=0.8))
    
    # Add index numbers
    for i in range(batch_size):
        ax.annotate(f'V{i}', 
                   xy=(target_2d[i, 0], target_2d[i, 1]),
                   xytext=(0, -25), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue',
                   ha='center')
    
    method = "PCA" if use_pca else f"t-SNE (perplexity={perplexity})"
    ax.set_title(f'2D Visualization of Embeddings with Videos ({method})', fontsize=16)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return target_2d, predicted_2d, fig
def create_sample_video_data():
    """Create sample embeddings and video tensors"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 4
    embedding_dim = 512
    
    # Create sample embeddings
    target_embeddings = np.random.randn(batch_size, embedding_dim) * 0.5
    predicted_embeddings = target_embeddings + np.random.randn(batch_size, embedding_dim) * 0.3
    
    # Create sample video tensors (list of videos, each with different frame counts)
    video_tensors_list = []
    
    # Video 1: 5 frames
    video1 = torch.randn(5, 3, 224, 224) * 0.5 + 0.5
    video1 = torch.clamp(video1, 0, 1)
    video_tensors_list.append(video1)
    
    # Video 2: 8 frames  
    video2 = torch.randn(8, 3, 224, 224) * 0.5 + 0.5
    video2 = torch.clamp(video2, 0, 1)
    video_tensors_list.append(video2)
    
    # Video 3: 6 frames
    video3 = torch.randn(6, 3, 224, 224) * 0.5 + 0.5
    video3 = torch.clamp(video3, 0, 1)
    video_tensors_list.append(video3)
    
    # Video 4: 10 frames
    video4 = torch.randn(10, 3, 224, 224) * 0.5 + 0.5
    video4 = torch.clamp(video4, 0, 1)
    video_tensors_list.append(video4)
    
    # Sample answer texts
    answer_texts = [
        "Sunset timelapse",
        "Cat playing with toy",
        "Flowers blooming",
        "City traffic flow"
    ]
    
    return target_embeddings, predicted_embeddings, video_tensors_list, answer_texts
def create_video_embedding_video_latest(target_embeddings, predicted_embeddings, 
                                video_tensors_list, answer_texts=None,
                                use_pca=True, random_state=42,
                                save_path="./video_embeddings.gif", 
                                fps_per_video=2, figsize=(14, 8)):
    """
    Create an animated video showing embeddings with videos playing frame by frame
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Video {i}" for i in range(batch_size)]
    
    # Process videos
    videos_np = process_video_tensors(video_tensors_list)
    
    # Reduce dimensionality
    all_embeddings = np.vstack([
            target_embeddings.detach().cpu().numpy(), 
            predicted_embeddings.detach().cpu().numpy()
        ])
    #all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Find maximum number of frames for timing
    max_frames = max([len(video) for video in videos_np])
    total_frames = sum([len(video) for video in videos_np])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Remove grid and axis numbers from both subplots
    ax1.grid(False)
    ax2.grid(False)
    
    # Set up embedding plot (show all points from start)
    target_scatter = ax1.scatter(
    target_2d[:, 0], target_2d[:, 1], 
    c='#3498db', marker='o', s=80, 
    label='Target', alpha=0.3, edgecolors='none'
    )
    predicted_scatter = ax1.scatter(
        predicted_2d[:, 0], predicted_2d[:, 1], 
        c='#95a5a6', marker='o', s=80, 
        label='Predicted', alpha=0.3, edgecolors='none'
    )

    # Current indicators (Same size, same shape, vibrant colors)
    current_target = ax1.scatter(
        [], [], 
        c='#00ff41', marker='o', s=80,  # Neon Green
        label='Current Video Target', 
        alpha=1.0, zorder=20, 
        edgecolors='white', linewidth=1.5
    )
    current_predicted = ax1.scatter(
        [], [], 
        c='#ff3f34', marker='o', s=80,  # Electric Red/Orange
        label='Current Video Predicted', 
        alpha=1.0, zorder=20, 
        edgecolors='white', linewidth=1.5
    )

    # Set limits with more padding for text
    all_points = np.vstack([target_2d, predicted_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.15  # Increased padding
    y_padding = (y_max - y_min) * 0.15  # Increased padding
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Remove axis numbers and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Remove spines (borders)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax1.set_title(f'Video Embedding Space ({method})', fontsize=14, pad=20)
    ax1.legend(loc='best', fontsize=10)
    
    # Create text overlays for target embeddings - INITIALLY just show indices
    target_texts = []
    for i in range(batch_size):
        # Add text directly on top of target points
        text = ax1.text(target_2d[i, 0], target_2d[i, 1], 
                       f'T{i}', 
                       fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       color='white',
                       bbox=dict(boxstyle="circle,pad=0.3", 
                                facecolor="blue", 
                                alpha=0.8,
                                edgecolor='black',
                                linewidth=1))
        target_texts.append(text)
    
    # Create text overlays for predicted embeddings - INITIALLY just show indices
    predicted_texts = []
    for i in range(batch_size):
        # Add text directly on top of predicted points
        text = ax1.text(predicted_2d[i, 0], predicted_2d[i, 1], 
                       f'P{i}', 
                       fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       color='white',
                       bbox=dict(boxstyle="square,pad=0.3", 
                                facecolor="red", 
                                alpha=0.8,
                                edgecolor='black',
                                linewidth=1))
        predicted_texts.append(text)
    
    # Create a text element for showing answer text on current TARGET
    current_target_answer_text = ax1.text(0, 0, '', 
                                          fontsize=11, fontweight='bold',
                                          ha='center', va='bottom',
                                          color='green',
                                          bbox=dict(boxstyle="round,pad=0.4", 
                                                   facecolor="white", 
                                                   alpha=0.9,
                                                   edgecolor='green',
                                                   linewidth=2),
                                          visible=False,
                                          zorder=20)  # Highest zorder to be on top
    
    # ADD THIS: Create a text element for showing answer text on current PREDICTED
    current_predicted_answer_text = ax1.text(0, 0, '', 
                                             fontsize=11, fontweight='bold',
                                             ha='center', va='top',
                                             color='orange',
                                             bbox=dict(boxstyle="round,pad=0.4", 
                                                      facecolor="white", 
                                                      alpha=0.9,
                                                      edgecolor='orange',
                                                      linewidth=2),
                                             visible=False,
                                             zorder=20)  # Highest zorder to be on top
    
    # Set up video display - remove axis numbers and grid
    ax2.axis('off')
    video_display = ax2.imshow(np.ones((224, 224, 3)))
    
    # Remove title from video display (we'll show info in overlay)
    ax2.set_title('')
    
    # Text displays - overlay on video
    video_info = ax2.text(0.5, 0.98, '', transform=ax2.transAxes, 
                         ha='center', va='top', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.4", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    frame_info = ax2.text(0.5, 0.02, '', transform=ax2.transAxes, 
                         ha='center', va='bottom', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    # Add video number overlay in top-left
    video_idx_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                            ha='left', va='top', fontsize=12, fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor="red", 
                                     alpha=0.8,
                                     edgecolor='white',
                                     linewidth=1))
    
    # Update function for animation
    def update(frame):
        # Determine which video and frame to show
        video_idx = 0
        frame_idx = 0
        cumulative_frames = 0
        
        for i in range(batch_size):
            if frame < cumulative_frames + len(videos_np[i]):
                video_idx = i
                frame_idx = frame - cumulative_frames
                break
            cumulative_frames += len(videos_np[i])
        
        # Update embedding highlights
        current_target.set_offsets([target_2d[video_idx]])
        current_predicted.set_offsets([predicted_2d[video_idx]])
        
        # Get answer text (truncate if too long)
        answer = answer_texts[video_idx]
        if len(answer) > 30:  # Adjust this threshold as needed
            display_answer = answer[:27] + "..."
        else:
            display_answer = answer
        
        # UPDATE 1: Show answer text on current TARGET (above the point)
        current_target_answer_text.set_position((target_2d[video_idx, 0], 
                                                target_2d[video_idx, 1] + y_padding * 0.08))
        current_target_answer_text.set_text(display_answer)
        current_target_answer_text.set_visible(True)
        
        # UPDATE 2: Show answer text on current PREDICTED (below the point)
        current_predicted_answer_text.set_position((predicted_2d[video_idx, 0], 
                                                   predicted_2d[video_idx, 1] - y_padding * 0.08))
        current_predicted_answer_text.set_text(display_answer)
        current_predicted_answer_text.set_visible(True)
        
        # Update video display
        ax2.clear()
        current_frame = videos_np[video_idx][frame_idx]
        ax2.imshow(current_frame)
        ax2.axis('off')
        
        # Update text overlays on video
        video_info.set_text(f'{answer_texts[video_idx]}')
        frame_info.set_text(f'Frame {frame_idx+1}/{len(videos_np[video_idx])}')
        video_idx_text.set_text(f'Video {video_idx}')
        
        # Add progress bar at bottom
        progress = (frame_idx + 1) / len(videos_np[video_idx])
        ax2.plot([0.1, 0.1 + 0.8 * progress], [0.05, 0.05], 
                color='yellow', linewidth=4, transform=ax2.transAxes)
        
        # Add frame number overlay in bottom-right
        ax2.text(0.95, 0.05, f'{frame_idx+1}/{len(videos_np[video_idx])}', 
                transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=10, fontweight='bold',
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="green", 
                         alpha=0.7,
                         edgecolor='white',
                         linewidth=1))
        
        # Update the target text overlays (make current one more prominent)
        for i, text in enumerate(target_texts):
            if i == video_idx:
                # For current video, replace "T{i}" with just the number
                text.set_text(f'{i}')
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.4", 
                                  facecolor="darkblue", 
                                  alpha=0.9,
                                  edgecolor='yellow',
                                  linewidth=2))
            else:
                # For other videos, show "T{i}"
                text.set_text(f'T{i}')
                text.set_fontsize(10)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.3", 
                                  facecolor="blue", 
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1))
        
        # Update the predicted text overlays (make current one more prominent)
        for i, text in enumerate(predicted_texts):
            if i == video_idx:
                # For current video, replace "P{i}" with just the number
                text.set_text(f'{i}')
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="square,pad=0.4", 
                                  facecolor="darkred", 
                                  alpha=0.9,
                                  edgecolor='yellow',
                                  linewidth=2))
            else:
                # For other videos, show "P{i}"
                text.set_text(f'P{i}')
                text.set_fontsize(10)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="square,pad=0.3", 
                                  facecolor="red", 
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1))
        
        return [target_scatter, predicted_scatter, current_target, 
                current_predicted, video_display, video_info, frame_info, 
                video_idx_text, current_target_answer_text, 
                current_predicted_answer_text] + target_texts + predicted_texts
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=1000//fps_per_video, blit=True)
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps_per_video)
        else:
            save_path = save_path + '.gif'
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        
        print(f"Video animation saved to: {save_path}")
        print(f"Total frames: {total_frames}")
        print(f"Video lengths: {[len(v) for v in videos_np]}")
    
    return ani
def create_video_embedding_video(target_embeddings, predicted_embeddings, 
                                video_tensors_list, answer_texts=None,
                                use_pca=True, random_state=42,
                                save_path="./video_embeddings.gif", 
                                fps_per_video=2, figsize=(14, 8)):
    """
    Create an animated video showing embeddings with videos playing frame by frame
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Video {i}" for i in range(batch_size)]
    
    # Process videos
    videos_np = process_video_tensors(video_tensors_list)
    
    # Reduce dimensionality
    all_embeddings = np.vstack([
            target_embeddings.detach().cpu().numpy(), 
            predicted_embeddings.detach().cpu().numpy()
        ])
    
    #all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Find maximum number of frames for timing
    max_frames = max([len(video) for video in videos_np])
    total_frames = sum([len(video) for video in videos_np])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Remove grid and axis numbers from both subplots
    ax1.grid(False)
    ax2.grid(False)
    
    # Set up embedding plot (show all points from start)
    # Background data: Use a softer palette and smaller markers to avoid clutter
    # Background data
    target_scatter = ax1.scatter(
        target_2d[:, 0], target_2d[:, 1], 
        c='#3498db', marker='o', s=80, 
        label='Target', alpha=0.3, edgecolors='none'
    )
    predicted_scatter = ax1.scatter(
        predicted_2d[:, 0], predicted_2d[:, 1], 
        c='#95a5a6', marker='o', s=80, 
        label='Predicted', alpha=0.3, edgecolors='none'
    )

    # Current indicators (Same size, same shape, vibrant colors)
    current_target = ax1.scatter(
        [], [], 
        c='#00ff41', marker='o', s=80,  # Neon Green
        label='Current Video Target', 
        alpha=1.0, zorder=20, 
        edgecolors='white', linewidth=1.5
    )
    current_predicted = ax1.scatter(
        [], [], 
        c='#ff3f34', marker='o', s=80,  # Electric Red/Orange
        label='Current Video Predicted', 
        alpha=1.0, zorder=20, 
        edgecolors='white', linewidth=1.5
    )
    
    # Set limits with more padding for text
    all_points = np.vstack([target_2d, predicted_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.15  # Increased padding
    y_padding = (y_max - y_min) * 0.15  # Increased padding
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Remove axis numbers and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Remove spines (borders)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax1.set_title(f'Embedding Space', fontsize=14, pad=20)
    ax1.legend(loc='best', fontsize=10)
    
    # Create text overlays for target embeddings - INITIALLY just show indices
    target_texts = []
    for i in range(batch_size):
        # Add text directly on top of target points
        text = ax1.text(target_2d[i, 0], target_2d[i, 1], 
                       f'T{i}', 
                       fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       color='white',
                       bbox=dict(boxstyle="circle,pad=0.3", 
                                facecolor="blue", 
                                alpha=0.8,
                                edgecolor='black',
                                linewidth=1))
        target_texts.append(text)
    
    # ADD THIS: Create a text element for showing answer text on current target
    current_target_answer_text = ax1.text(0, 0, '', 
                                          fontsize=11, fontweight='bold',
                                          ha='center', va='bottom',
                                          color='green',
                                          bbox=dict(boxstyle="round,pad=0.4", 
                                                   facecolor="white", 
                                                   alpha=0.9,
                                                   edgecolor='green',
                                                   linewidth=2),
                                          visible=False,
                                          zorder=20)  # Highest zorder to be on top
    
    # Set up video display - remove axis numbers and grid
    ax2.axis('off')
    video_display = ax2.imshow(np.ones((224, 224, 3)))
    
    # Remove title from video display (we'll show info in overlay)
    ax2.set_title('')
    
    # Text displays - overlay on video
    video_info = ax2.text(0.5, 0.98, '', transform=ax2.transAxes, 
                         ha='center', va='top', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.4", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    frame_info = ax2.text(0.5, 0.02, '', transform=ax2.transAxes, 
                         ha='center', va='bottom', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    # Add video number overlay in top-left
    video_idx_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                            ha='left', va='top', fontsize=12, fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor="red", 
                                     alpha=0.8,
                                     edgecolor='white',
                                     linewidth=1))
    
    # Update function for animation
    def update(frame):
        # Determine which video and frame to show
        video_idx = 0
        frame_idx = 0
        cumulative_frames = 0
        
        for i in range(batch_size):
            if frame < cumulative_frames + len(videos_np[i]):
                video_idx = i
                frame_idx = frame - cumulative_frames
                break
            cumulative_frames += len(videos_np[i])
        
        # Update embedding highlights
        current_target.set_offsets([target_2d[video_idx]])
        current_predicted.set_offsets([predicted_2d[video_idx]])
        
        # UPDATE: Show answer text on current target
        # Position the answer text above the current target
        current_target_answer_text.set_position((target_2d[video_idx, 0], 
                                                target_2d[video_idx, 1] + y_padding * 0.1))
        
        # Truncate answer text if too long
        answer = answer_texts[video_idx]
        if len(answer) > 30:  # Adjust this threshold as needed
            display_answer = answer[:27] + "..."
        else:
            display_answer = answer
            
        current_target_answer_text.set_text(display_answer)
        current_target_answer_text.set_visible(True)
        
        # Update video display
        ax2.clear()
        current_frame = videos_np[video_idx][frame_idx]
        ax2.imshow(current_frame)
        ax2.axis('off')
        
        # Update text overlays on video
        video_info.set_text(f'{answer_texts[video_idx]}')
        frame_info.set_text(f'Frame {frame_idx+1}/{len(videos_np[video_idx])}')
        video_idx_text.set_text(f'Video {video_idx}')
        
        # Add progress bar at bottom
        progress = (frame_idx + 1) / len(videos_np[video_idx])
        ax2.plot([0.1, 0.1 + 0.8 * progress], [0.05, 0.05], 
                color='yellow', linewidth=4, transform=ax2.transAxes)
        
        # Add frame number overlay in bottom-right
        ax2.text(0.95, 0.05, f'{frame_idx+1}/{len(videos_np[video_idx])}', 
                transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=10, fontweight='bold',
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="green", 
                         alpha=0.7,
                         edgecolor='white',
                         linewidth=1))
        
        # Update the target text overlays (make current one more prominent)
        for i, text in enumerate(target_texts):
            if i == video_idx:
                # For current video, replace "T{i}" with just the number
                text.set_text(f'{i}')
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.4", 
                                  facecolor="darkblue", 
                                  alpha=0.9,
                                  edgecolor='yellow',
                                  linewidth=2))
            else:
                # For other videos, show "T{i}"
                text.set_text(f'T{i}')
                text.set_fontsize(10)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.3", 
                                  facecolor="blue", 
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1))
        
        return [target_scatter, predicted_scatter, current_target, 
                current_predicted, video_display, video_info, frame_info, 
                video_idx_text, current_target_answer_text] + target_texts
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=1000//fps_per_video, blit=True)
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps_per_video)
        else:
            save_path = save_path + '.gif'
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        
        print(f"Video animation saved to: {save_path}")
        print(f"Total frames: {total_frames}")
        print(f"Video lengths: {[len(v) for v in videos_np]}")
    
    return ani


def create_video_embedding_vide_2(target_embeddings, predicted_embeddings, 
                                video_tensors_list, answer_texts=None,
                                use_pca=True, random_state=42,
                                save_path="./video_embeddings.gif", 
                                fps_per_video=2, figsize=(14, 8)):
    """
    Create an animated video showing embeddings with videos playing frame by frame
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Video {i}" for i in range(batch_size)]
    
    # Process videos
    videos_np = process_video_tensors(video_tensors_list)
    
    # Reduce dimensionality
    all_embeddings = np.vstack([
            target_embeddings.detach().cpu().numpy(), 
            predicted_embeddings.detach().cpu().numpy()
        ])
    # all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Find maximum number of frames for timing
    max_frames = max([len(video) for video in videos_np])
    total_frames = sum([len(video) for video in videos_np])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Remove grid and axis numbers from both subplots
    ax1.grid(False)
    ax2.grid(False)
    
    # Set up embedding plot (show all points from start)
    target_scatter = ax1.scatter(target_2d[:, 0], target_2d[:, 1], 
                                 c='blue', marker='o', s=120, 
                                 label='Target', alpha=0.8, edgecolors='black', linewidth=1.5)
    predicted_scatter = ax1.scatter(predicted_2d[:, 0], predicted_2d[:, 1], 
                                    c='red', marker='s', s=120, 
                                    label='Predicted', alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Current video indicator
    current_target = ax1.scatter([], [], c='green', marker='*', s=250, 
                                 label='Current Video Target', alpha=1.0, zorder=10)
    current_predicted = ax1.scatter([], [], c='orange', marker='*', s=250, 
                                    label='Current Video Predicted', alpha=1.0, zorder=10)
    
    # Set limits with more padding for text
    all_points = np.vstack([target_2d, predicted_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.15  # Increased padding
    y_padding = (y_max - y_min) * 0.15  # Increased padding
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Remove axis numbers and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Remove spines (borders)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax1.set_title(f'Video Embedding Space ({method})', fontsize=14, pad=20)
    ax1.legend(loc='best', fontsize=10)
    
    # Create text overlays for target embeddings
    target_texts = []
    for i in range(batch_size):
        # Add text directly on top of target points
        text = ax1.text(target_2d[i, 0], target_2d[i, 1], 
                       f'T{i}', 
                       fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       color='white',
                       bbox=dict(boxstyle="circle,pad=0.3", 
                                facecolor="blue", 
                                alpha=0.8,
                                edgecolor='black',
                                linewidth=1))
        target_texts.append(text)
    
    # Set up video display - remove axis numbers and grid
    ax2.axis('off')
    video_display = ax2.imshow(np.ones((224, 224, 3)))
    
    # Remove title from video display (we'll show info in overlay)
    ax2.set_title('')
    
    # Text displays - overlay on video
    video_info = ax2.text(0.5, 0.98, '', transform=ax2.transAxes, 
                         ha='center', va='top', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.4", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    frame_info = ax2.text(0.5, 0.02, '', transform=ax2.transAxes, 
                         ha='center', va='bottom', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="black", 
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))
    
    # Add video number overlay in top-left
    video_idx_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                            ha='left', va='top', fontsize=12, fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor="red", 
                                     alpha=0.8,
                                     edgecolor='white',
                                     linewidth=1))
    
    # Update function for animation
    def update(frame):
        # Determine which video and frame to show
        video_idx = 0
        frame_idx = 0
        cumulative_frames = 0
        
        for i in range(batch_size):
            if frame < cumulative_frames + len(videos_np[i]):
                video_idx = i
                frame_idx = frame - cumulative_frames
                break
            cumulative_frames += len(videos_np[i])
        
        # Update embedding highlights
        current_target.set_offsets([target_2d[video_idx]])
        current_predicted.set_offsets([predicted_2d[video_idx]])
        
        # Update video display
        ax2.clear()
        current_frame = videos_np[video_idx][frame_idx]
        ax2.imshow(current_frame)
        ax2.axis('off')
        
        # Update text overlays on video
        video_info.set_text(f'{answer_texts[video_idx]}')
        frame_info.set_text(f'Frame {frame_idx+1}/{len(videos_np[video_idx])}')
        video_idx_text.set_text(f'Video {video_idx}')
        
        # Add progress bar at bottom
        progress = (frame_idx + 1) / len(videos_np[video_idx])
        ax2.plot([0.1, 0.1 + 0.8 * progress], [0.05, 0.05], 
                color='yellow', linewidth=4, transform=ax2.transAxes)
        
        # Add frame number overlay in bottom-right
        ax2.text(0.95, 0.05, f'{frame_idx+1}/{len(videos_np[video_idx])}', 
                transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=10, fontweight='bold',
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="green", 
                         alpha=0.7,
                         edgecolor='white',
                         linewidth=1))
        
        # Update the target text overlays (make current one more prominent)
        for i, text in enumerate(target_texts):
            if i == video_idx:
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.4", 
                                  facecolor="darkblue", 
                                  alpha=0.9,
                                  edgecolor='yellow',
                                  linewidth=2))
            else:
                text.set_fontsize(10)
                text.set_fontweight('bold')
                text.set_bbox(dict(boxstyle="circle,pad=0.3", 
                                  facecolor="blue", 
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1))
        
        return [target_scatter, predicted_scatter, current_target, 
                current_predicted, video_display, video_info, frame_info, 
                video_idx_text] + target_texts
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=1000//fps_per_video, blit=True)
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps_per_video)
        else:
            save_path = save_path + '.gif'
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        
        print(f"Video animation saved to: {save_path}")
        print(f"Total frames: {total_frames}")
        print(f"Video lengths: {[len(v) for v in videos_np]}")
    
    return ani





def create_video_embedding_video_old(target_embeddings, predicted_embeddings, 
                                video_tensors_list, answer_texts=None,
                                use_pca=True, random_state=42,
                                save_path="./video_embeddings.gif", 
                                fps_per_video=2, figsize=(14, 8)):
    """
    Create an animated video showing embeddings with videos playing frame by frame
    """
    
    batch_size = target_embeddings.shape[0]
    
    if answer_texts is None:
        answer_texts = [f"Video {i}" for i in range(batch_size)]
    
    # Process videos
    videos_np = process_video_tensors(video_tensors_list)
    
    # Reduce dimensionality
    #all_embeddings = np.vstack([target_embeddings, predicted_embeddings])
    all_embeddings = np.vstack([
            target_embeddings.detach().cpu().numpy(), 
            predicted_embeddings.detach().cpu().numpy()
        ])
    if use_pca:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "PCA"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method = "t-SNE"
    
    target_2d = reduced_embeddings[:batch_size]
    predicted_2d = reduced_embeddings[batch_size:]
    
    # Find maximum number of frames for timing
    max_frames = max([len(video) for video in videos_np])
    total_frames = sum([len(video) for video in videos_np])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set up embedding plot (show all points from start)
    target_scatter = ax1.scatter(target_2d[:, 0], target_2d[:, 1], 
                                 c='green', marker='o', s=60, 
                                 label='Target', alpha=0.7, edgecolors='black')
    predicted_scatter = ax1.scatter(predicted_2d[:, 0], predicted_2d[:, 1], 
                                    c='yellow', marker='o', s=60, 
                                    label='Predicted', alpha=0.7, edgecolors='black')
    
    # Current video indicator
    current_target = ax1.scatter([], [], c='blue', marker='*', s=90, 
                                 label='Current Video Target', alpha=1.0)
    current_predicted = ax1.scatter([], [], c='orange', marker='*', s=90, 
                                    label='Current Video Predicted', alpha=1.0)
    
    # Set limits
    all_points = np.vstack([target_2d, predicted_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_title(f'Video Embedding Space ({method})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Set up video display
    ax2.axis('off')
    video_display = ax2.imshow(np.ones((224, 224, 3)))
    ax2.set_title('Video Frame Display')
    
    # Text displays
    video_info = ax2.text(0.5, -0.05, '', transform=ax2.transAxes, 
                         ha='center', va='top', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    frame_info = ax2.text(0.5, -0.15, '', transform=ax2.transAxes, 
                         ha='center', va='top', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Update function for animation
    def update(frame):
        # Determine which video and frame to show
        video_idx = 0
        frame_idx = 0
        cumulative_frames = 0
        
        for i in range(batch_size):
            if frame < cumulative_frames + len(videos_np[i]):
                video_idx = i
                frame_idx = frame - cumulative_frames
                break
            cumulative_frames += len(videos_np[i])
        
        # Update embedding highlights
        current_target.set_offsets([target_2d[video_idx]])
        current_predicted.set_offsets([predicted_2d[video_idx]])
        
        # Update video display
        ax2.clear()
        current_frame = videos_np[video_idx][frame_idx]
        ax2.imshow(current_frame)
        ax2.axis('off')
        
        # Update text info
        video_info.set_text(f'Video {video_idx}: {answer_texts[video_idx]}')
        frame_info.set_text(f'Frame {frame_idx+1}/{len(videos_np[video_idx])}')
        
        # Add progress bar
        progress = (frame_idx + 1) / len(videos_np[video_idx])
        ax2.set_title(f'Video {video_idx} - Frame {frame_idx+1}/{len(videos_np[video_idx])}')
        
        # Draw progress bar
        ax2.plot([0.1, 0.1 + 0.8 * progress], [0.95, 0.95], 
                color='red', linewidth=3, transform=ax2.transAxes)
        
        return [target_scatter, predicted_scatter, current_target, 
                current_predicted, video_display, video_info, frame_info]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=1000//fps_per_video, blit=True)
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps_per_video)
        else:
            save_path = save_path + '.gif'
            ani.save(save_path, writer='pillow', fps=fps_per_video, dpi=100)
        
        print(f"Video animation saved to: {save_path}")
        print(f"Total frames: {total_frames}")
        print(f"Video lengths: {[len(v) for v in videos_np]}")
    
    return ani


# Main execution
if __name__ == "__main__":
    # Create sample video data
    print("Generating sample video data...")
    target_emb, pred_emb, video_tensors, texts = create_sample_video_data()
    
    print("\nData shapes:")
    print(f"Target embeddings: {target_emb.shape}")
    print(f"Predicted embeddings: {pred_emb.shape}")
    print(f"Number of videos: {len(video_tensors)}")
    for i, video in enumerate(video_tensors):
        print(f"  Video {i}: {video.shape} (frames, channels, height, width)")
    print(f"Video descriptions: {texts}")
    print("-" * 50)
    
    # Create results directory
    results_dir = "./video_embedding_visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Option 2: Video animation
    print("\n2. Creating video animation...")
    ani_seq = create_video_embedding_video(
        target_emb, pred_emb, video_tensors, texts,
        use_pca=True,
        save_path=f"{results_dir}/video_animation_{timestamp}.gif",
        fps_per_video=5,
        figsize=(14, 7)
    )
    print(f"  - video_animation_{timestamp}.gif (Video animation)")
    print(f"  - *.npy files (Embeddings data)")
# # Main execution
# if __name__ == "__main__":
#     # Create sample data
#     print("Generating sample data...")
#     target_emb, pred_emb, image_tensors, texts = create_sample_image_data()
    
#     print("\nData shapes:")
#     print(f"Target embeddings: {target_emb.shape}")
#     print(f"Predicted embeddings: {pred_emb.shape}")
#     print(f"Image tensors: {image_tensors.shape}")
#     print(f"Number of samples: {len(texts)}")
#     print("-" * 50)
    
#     # Create results directory
#     results_dir = "./embedding_visualizations_with_images"
#     os.makedirs(results_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Option 1: Static visualization with images
#     print("\nCreating static visualization with images...")
#     target_2d, pred_2d, fig = visualize_embeddings_with_images(
#         target_emb, pred_emb, image_tensors, texts,
#         use_pca=True,
#         save_path=f"{results_dir}/embeddings_with_images_{timestamp}.png",
#         dpi=150,
#         figsize=(16, 12)
#     )
#     plt.show()
    
#     # Option 2: Create animated video
#     print("\nCreating animated video...")
#     ani = create_embedding_video(
#         target_emb, pred_emb, image_tensors, texts,
#         use_pca=True,
#         save_path=f"{results_dir}/embedding_video_{timestamp}.gif",
#         fps=1,  # 1 frame per second
#         figsize=(14, 7)
#     )
    
#     # Display animation in notebook (if in Jupyter)
#     try:
#         from IPython.display import HTML
#         HTML(ani.to_jshtml())
#     except:
#         print("Animation created. Check saved GIF file.")
    
#     # Option 3: Interactive visualization with image hover
#     print("\nCreating interactive visualization...")
#     fig_interactive = visualize_embeddings_interactive_with_images(
#         target_emb, pred_emb, image_tensors, texts,
#         use_pca=True,
#         save_path=f"{results_dir}/interactive_embeddings_{timestamp}.html",
#         width=1400,
#         height=700
#     )
    
#     # Additional analysis
#     print("\nDistance and similarity metrics:")
#     for i in range(len(texts)):
#         distance = np.linalg.norm(target_emb[i] - pred_emb[i])
#         cos_sim = np.dot(target_emb[i], pred_emb[i]) / (
#             np.linalg.norm(target_emb[i]) * np.linalg.norm(pred_emb[i])
#         )
#         print(f"Sample {i} ({texts[i][:30]}...):")
#         print(f"  Euclidean distance: {distance:.4f}")
#         print(f"  Cosine similarity: {cos_sim:.4f}")
    
#     # Save data
#     print("\nSaving data...")
#     np.save(f"{results_dir}/target_embeddings_{timestamp}.npy", target_emb)
#     np.save(f"{results_dir}/predicted_embeddings_{timestamp}.npy", pred_emb)
#     np.save(f"{results_dir}/target_2d_{timestamp}.npy", target_2d)
#     np.save(f"{results_dir}/predicted_2d_{timestamp}.npy", pred_2d)
    
#     print(f"\nAll visualizations saved in: {results_dir}")
#     print(f"Files created:")
#     print(f"  - embeddings_with_images_{timestamp}.png (Static visualization)")
#     print(f"  - embedding_video_{timestamp}.gif (Animated GIF)")
#     print(f"  - interactive_embeddings_{timestamp}.html (Interactive plot)")
#     print(f"  - *.npy files (Embeddings data)") #minor change the images are changed per batch can it be a video change per batch because the batch of images is a list of videos of different no of images