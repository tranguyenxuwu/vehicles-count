#!/usr/bin/env python3
"""
Script to split YOLO dataset into train and validation sets.
This script takes the existing train dataset and splits it into separate
train and validation folders while maintaining the image-label pairs.
"""

import os
import shutil
import random
from pathlib import Path

def create_directories(base_path, folders):
    """
    Create necessary directories for the split dataset.
    
    Args:
        base_path (str): Base path where directories will be created
        folders (list): List of folder names to create
    """
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created directory: {folder_path}")

def get_image_label_pairs(images_dir, labels_dir):
    """
    Get matching image-label pairs from the directories.
    
    Args:
        images_dir (str): Path to images directory
        labels_dir (str): Path to labels directory
    
    Returns:
        list: List of tuples (image_path, label_path) for matching pairs
    """
    image_files = {}
    label_files = {}
    
    # Get all image files
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in Path(images_dir).glob(ext):
            # Use stem (filename without extension) as key
            base_name = img_path.stem
            image_files[base_name] = str(img_path)
    
    # Get all label files
    for label_path in Path(labels_dir).glob('*.txt'):
        base_name = label_path.stem
        label_files[base_name] = str(label_path)
    
    # Find matching pairs
    pairs = []
    for base_name in image_files:
        if base_name in label_files:
            pairs.append((image_files[base_name], label_files[base_name]))
        else:
            print(f"Warning: No label file found for image: {base_name}")
    
    # Check for labels without images
    for base_name in label_files:
        if base_name not in image_files:
            print(f"Warning: No image file found for label: {base_name}")
    
    return pairs

def split_dataset(source_dir, output_dir, val_ratio=0.2, random_seed=42):
    """
    Split dataset into train and validation sets.
    
    Args:
        source_dir (str): Path to source directory containing train folder
        output_dir (str): Path to output directory for split dataset
        val_ratio (float): Ratio of validation data (0.0 to 1.0)
        random_seed (int): Random seed for reproducible splits
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Define paths
    train_images_dir = os.path.join(source_dir, 'train', 'images')
    train_labels_dir = os.path.join(source_dir, 'train', 'labels')
    
    # Check if source directories exist
    if not os.path.exists(train_images_dir):
        print(f"Error: Source images directory not found: {train_images_dir}")
        return
    
    if not os.path.exists(train_labels_dir):
        print(f"Error: Source labels directory not found: {train_labels_dir}")
        return
    
    # Create output directories
    dirs_to_create = [
        'train/images', 'train/labels',
        'val/images', 'val/labels'
    ]
    
    for directory in dirs_to_create:
        create_directories(output_dir, [directory])
    
    # Get image-label pairs
    print("Finding image-label pairs...")
    pairs = get_image_label_pairs(train_images_dir, train_labels_dir)
    
    if not pairs:
        print("Error: No matching image-label pairs found!")
        return
    
    print(f"Found {len(pairs)} image-label pairs")
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Calculate split sizes
    total_pairs = len(pairs)
    val_size = int(total_pairs * val_ratio)
    train_size = total_pairs - val_size
    
    print(f"Splitting into:")
    print(f"  Training: {train_size} pairs ({(1-val_ratio)*100:.1f}%)")
    print(f"  Validation: {val_size} pairs ({val_ratio*100:.1f}%)")
    
    # Split the data
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    
    # Copy files to respective directories
    print("\nCopying training files...")
    for i, (img_path, label_path) in enumerate(train_pairs):
        # Copy image
        dst_img = os.path.join(output_dir, 'train', 'images', os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        dst_label = os.path.join(output_dir, 'train', 'labels', os.path.basename(label_path))
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(train_pairs)} training pairs")
    
    print("\nCopying validation files...")
    for i, (img_path, label_path) in enumerate(val_pairs):
        # Copy image
        dst_img = os.path.join(output_dir, 'val', 'images', os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        dst_label = os.path.join(output_dir, 'val', 'labels', os.path.basename(label_path))
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(val_pairs)} validation pairs")
    
    print(f"\nDataset split completed!")
    print(f"Output directory: {output_dir}")
    print(f"Training set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

def main():
    """Main function to run the dataset splitting."""
    # Configuration
    current_dir = os.getcwd()
    source_dir = current_dir  # Current directory contains the train folder
    output_dir = os.path.join(current_dir, 'split_dataset')  # Output to split_dataset folder
    val_ratio = 0.2  # 20% for validation
    random_seed = 42  # For reproducible results
    
    print("YOLO Dataset Splitter")
    print("====================")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Validation ratio: {val_ratio * 100:.1f}%")
    print(f"Random seed: {random_seed}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the split? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Perform the split
    split_dataset(source_dir, output_dir, val_ratio, random_seed)
    
    print("\nNote: The original train folder remains unchanged.")
    print("The split dataset is available in the 'split_dataset' folder.")

if __name__ == "__main__":
    main()
