import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from google.colab import drive

# Milestone 1: AI Oil Spill Detection Preprocessing Pipeline with Google Drive Integration

def mount_drive_and_get_dataset_path():
    """
    Mount Google Drive on Colab and return the folder path to the dataset.
    """
    drive.mount('/content/drive')
    # Change this to your Google Drive folder path holding dataset
    dataset_folder = '/content/drive/MyDrive/1NaBM5eigrR0oz6N82mtBj56G3ZzQ7PXx'
    return dataset_folder

def load_images_from_folder(folder_path, img_size=(256, 256)):
    """
    Load images from the specified folder path, resize to img_size,
    and convert to grayscale (SAR images tend to be grayscale).
    """
    images = []
    image_paths = glob(os.path.join(folder_path, '*.*'))
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        images.append(img)
    return images

def visualize_samples(images, n=5):
    """Visualize n sample images from the dataset."""
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Sample {i+1}')
        plt.axis('off')
    plt.show()

def normalize_images(images):
    """Normalize image pixel values to range [0, 1]."""
    norm_images = [img / 255.0 for img in images]
    return norm_images

def speckle_noise_reduction(images):
    """
    Apply median blur filtering to reduce speckle noise common in SAR images.
    """
    denoised = [cv2.medianBlur((img * 255).astype(np.uint8), ksize=3) / 255.0 for img in images]
    return denoised

def augment_images(images):
    """
    Extra Feature: Data augmentation via flipping, rotations,
    brightness and contrast adjustments to improve model robustness.
    """
    augmented_images = []
    for img in images:
        augmented_images.append(img)  # Original

        # Horizontal flip
        flipped_h = cv2.flip((img * 255).astype(np.uint8), 1) / 255.0
        augmented_images.append(flipped_h)

        # Vertical flip
        flipped_v = cv2.flip((img * 255).astype(np.uint8), 0) / 255.0
        augmented_images.append(flipped_v)

        # Rotation 90 degrees clockwise
        rotated = cv2.rotate((img * 255).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE) / 255.0
        augmented_images.append(rotated)

        # Brightness increase (extra)
        bright = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=1.2, beta=30) / 255.0
        augmented_images.append(bright)

        # Brightness decrease (extra)
        dark = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=0.8, beta=-20) / 255.0
        augmented_images.append(dark)

        # Contrast adjustment (extra)
        contrasted = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=1.5, beta=0) / 255.0
        augmented_images.append(contrasted)

    return augmented_images

def compute_statistics(images):
    """Compute and print mean and standard deviation of dataset pixel values."""
    means = [np.mean(img) for img in images]
    stds = [np.std(img) for img in images]
    print(f"Dataset Mean Pixel Value: {np.mean(means):.4f}")
    print(f"Dataset Pixel Value Std Dev: {np.mean(stds):.4f}")

def split_dataset(images, test_size=0.2, val_size=0.1):
    """Split images into training, validation, and test sets."""
    train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)
    return train_imgs, val_imgs, test_imgs

def save_datasets(train_imgs, val_imgs, test_imgs):
    """Save processed datasets as .npy files for later milestones."""
    np.save('train_images.npy', np.array(train_imgs))
    np.save('val_images.npy', np.array(val_imgs))
    np.save('test_images.npy', np.array(test_imgs))
    print("Preprocessed datasets saved as .npy files.")

# Main
if __name__ == "__main__":
    # Mount Google Drive and get dataset folder path
    dataset_folder = mount_drive_and_get_dataset_path()

    # Load raw images
    raw_images = load_images_from_folder(dataset_folder)
    print(f"Loaded {len(raw_images)} raw images")

    # Visualize example raw images
    print("Visualizing raw images...")
    visualize_samples(raw_images, n=5)

    # Normalize images to [0,1]
    norm_images = normalize_images(raw_images)

    # Denoise speckle noise specific to SAR images
    denoised_images = speckle_noise_reduction(norm_images)

    # Visualize denoised images
    print("Visualizing denoised images...")
    visualize_samples(denoised_images, n=5)

    # Compute and show statistics before augmentation
    print("Statistics before augmentation:")
    compute_statistics(denoised_images)

    # Augmentation is extra but makes dataset robust
    augmented_images = augment_images(denoised_images)
    print(f"Dataset size after augmentation: {len(augmented_images)}")

    # Visualize augmented images
    print("Visualizing augmented images...")
    visualize_samples(augmented_images, n=8)

    # Statistics after augmentation
    print("Statistics after augmentation:")
    compute_statistics(augmented_images)

    # Split dataset for training, validation, and testing
    train_imgs, val_imgs, test_imgs = split_dataset(augmented_images)
    print(f"Training set size: {len(train_imgs)}")
    print(f"Validation set size: {len(val_imgs)}")
    print(f"Testing set size: {len(test_imgs)}")

    # Save datasets for later milestones
    save_datasets(train_imgs, val_imgs, test_imgs)
