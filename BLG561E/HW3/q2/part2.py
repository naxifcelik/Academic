import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import os

# Load the StyleGAN3 model
with open("/home/nax/Desktop/hw3/stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    a = pickle.load(f)

gan = a["G_ema"]
gan.eval()

# Disable gradients for inference
for param in gan.parameters():
    param.requires_grad = False

def generate_image(z_vector):
    """Generate image from latent vector"""
    with torch.no_grad():
        img = gan(z_vector, 0).numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, -1, 1)  # Clamp values
        img = 255 * (img + 1) / 2  # Convert to 0-255 range
        img = img.astype(np.uint8)
    return img[:, :, [2, 1, 0]]  # Convert RGB to BGR for OpenCV

def interpolate_vectors(z1, z2, steps=50):
    """Create interpolation between two latent vectors"""
    interpolated = []
    for i in range(steps + 1):
        alpha = i / steps
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z_interp)
    return interpolated

# Create two random latent vectors
torch.manual_seed(42)  # For reproducible results
z1 = torch.randn(1, 512)
torch.manual_seed(123)
z2 = torch.randn(1, 512)

print("Generating first face...")
img1 = generate_image(z1)
cv2.imwrite('face1.png', img1)

print("Generating second face...")
img2 = generate_image(z2)
cv2.imwrite('face2.png', img2)

# Create interpolation
print("Creating interpolation frames...")
num_steps = 50  # Number of frames in the animation
interpolated_vectors = interpolate_vectors(z1, z2, num_steps)

# Create output directory for frames
os.makedirs('animation_frames', exist_ok=True)

# Generate all frames
frames = []
for i, z_interp in enumerate(interpolated_vectors):
    print(f"Generating frame {i+1}/{len(interpolated_vectors)}")
    img = generate_image(z_interp)
    
    # Save individual frame
    frame_path = f'animation_frames/frame_{i:03d}.png'
    cv2.imwrite(frame_path, img)
    frames.append(img)

print("Creating animation...")

# Create video using OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('face_morph_animation.mp4', fourcc, 10.0, (1024, 1024))

# Write frames to video
for frame in frames:
    video_writer.write(frame)

# Add some frames at the end to pause on final image
for _ in range(20):
    video_writer.write(frames[-1])

video_writer.release()

print("Animation saved as 'face_morph_animation.mp4'")
print(f"Generated {len(frames)} frames")
print("Individual frames saved in 'animation_frames/' directory")

# Optional: Create a GIF as well (requires pillow)
try:
    from PIL import Image
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb_frame))
    
    # Save as GIF
    pil_frames[0].save(
        'face_morph_animation.gif',
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    print("Animation also saved as 'face_morph_animation.gif'")
    
except ImportError:
    print("PIL not available - skipping GIF creation")

# Display some sample frames using matplotlib
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]

for i, idx in enumerate(sample_indices):
    # Convert BGR to RGB for matplotlib
    rgb_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
    axes[i].imshow(rgb_frame)
    axes[i].set_title(f'Frame {idx}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('sample_frames.png', dpi=150, bbox_inches='tight')
plt.show()

print("Sample frames visualization saved as 'sample_frames.png'")