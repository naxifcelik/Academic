import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from glob import glob

# Settings
input_dir = "Imagenet_50/train"
output_dir = "Imagenet_50_aug/train"
undersample_ratio = 0.5  # e.g., keep 50% of images randomly

# Create output directories
os.makedirs(output_dir, exist_ok=True)

def augment_image(img):
    h, w = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Flip horizontally
    flipped = cv2.flip(img, 1)

    # Rotate ±15 degrees
    angle = random.choice([-15, 15])
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    # Zoom (cropping and resizing)
    zoom_factor = 1.2
    center_x, center_y = w // 2, h // 2
    crop_size = int(min(w, h) / zoom_factor)
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    cropped = img[y1:y1 + crop_size, x1:x1 + crop_size]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return [gray, flipped, rotated, zoomed]

def process_class(class_path, class_name):
    out_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    images = glob(os.path.join(class_path, "*.*"))
    sampled_images = random.sample(images, max(1, int(len(images) * undersample_ratio)))

    for img_path in tqdm(sampled_images, desc=f"Processing {class_name}", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_class_dir, f"{base_name}_original.jpg"), img)

        aug_images = augment_image(img)
        aug_suffixes = ["grayscale", "flipped", "rotated", "zoomed"]

        for aug_img, suffix in zip(aug_images, aug_suffixes):
            aug_name = f"{base_name}_{suffix}.jpg"
            cv2.imwrite(os.path.join(out_class_dir, aug_name), aug_img)

def main():
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for class_name in tqdm(class_dirs, desc="Classes"):
        class_path = os.path.join(input_dir, class_name)
        process_class(class_path, class_name)

if __name__ == "__main__":
    main()
