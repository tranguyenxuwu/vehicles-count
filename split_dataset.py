import os
import shutil
import random
from pathlib import Path

def create_directories(base_path, folders):
    """
    Tạo các thư mục cần thiết cho dataset đã chia.
    
    Args:
        base_path (str): Đường dẫn gốc nơi tạo thư mục
        folders (list): Danh sách tên thư mục cần tạo
    """
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created directory: {folder_path}")

def get_image_label_pairs(images_dir, labels_dir):
    """
    Lấy các cặp ảnh-nhãn khớp nhau từ thư mục.
    
    Args:
        images_dir (str): Đường dẫn thư mục ảnh
        labels_dir (str): Đường dẫn thư mục nhãn
    
    Returns:
        list: Danh sách các tuple (image_path, label_path) cho các cặp khớp
    """
    image_files = {}
    label_files = {}
    
    # Lấy tất cả file ảnh với các định dạng phổ biến
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in Path(images_dir).glob(ext):
            # Sử dụng stem (tên file không có phần mở rộng) làm key
            base_name = img_path.stem
            image_files[base_name] = str(img_path)
    
    # Lấy tất cả file nhãn
    for label_path in Path(labels_dir).glob('*.txt'):
        base_name = label_path.stem
        label_files[base_name] = str(label_path)
    
    # Tìm các cặp khớp nhau
    pairs = []
    for base_name in image_files:
        if base_name in label_files:
            pairs.append((image_files[base_name], label_files[base_name]))
        else:
            print(f"Warning: No label file found for image: {base_name}")
    
    # Kiểm tra nhãn không có ảnh tương ứng
    for base_name in label_files:
        if base_name not in image_files:
            print(f"Warning: No image file found for label: {base_name}")
    
    return pairs

def split_dataset(source_dir, output_dir, val_ratio=0.2, random_seed=42):
    """
    Chia dataset thành tập train và validation.
    
    Args:
        source_dir (str): Đường dẫn đến thư mục nguồn chứa thư mục train
        output_dir (str): Đường dẫn đến thư mục đầu ra cho dataset đã chia
        val_ratio (float): Tỷ lệ dữ liệu validation (0.0 đến 1.0)
        random_seed (int): Hạt giống ngẫu nhiên cho phép chia có thể tái tạo
    """
    # Đặt hạt giống ngẫu nhiên cho khả năng tái tạo
    random.seed(random_seed)
    
    # Định nghĩa các đường dẫn
    train_images_dir = os.path.join(source_dir, 'train', 'images')
    train_labels_dir = os.path.join(source_dir, 'train', 'labels')
    
    # Kiểm tra xem các thư mục nguồn có tồn tại không
    if not os.path.exists(train_images_dir):
        print(f"Error: Source images directory not found: {train_images_dir}")
        return
    
    if not os.path.exists(train_labels_dir):
        print(f"Error: Source labels directory not found: {train_labels_dir}")
        return
    
    # Tạo các thư mục đầu ra
    dirs_to_create = [
        'train/images', 'train/labels',
        'val/images', 'val/labels'
    ]
    
    for directory in dirs_to_create:
        full_path = os.path.join(output_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    # Lấy các cặp ảnh-nhãn
    print("Finding image-label pairs...")
    pairs = get_image_label_pairs(train_images_dir, train_labels_dir)
    
    if not pairs:
        print("Error: No matching image-label pairs found!")
        return
    
    print(f"Found {len(pairs)} image-label pairs")
    
    # Xáo trộn các cặp
    random.shuffle(pairs)
    
    # Tính toán kích thước chia tách
    total_pairs = len(pairs)
    val_size = int(total_pairs * val_ratio)
    train_size = total_pairs - val_size
    
    print(f"Splitting into:")
    print(f"  Training: {train_size} pairs ({(1-val_ratio)*100:.1f}%)")
    print(f"  Validation: {val_size} pairs ({val_ratio*100:.1f}%)")
    
    # Chia dữ liệu
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    
    # Sao chép file vào các thư mục tương ứng
    print("\nCopying training files...")
    for i, (img_path, label_path) in enumerate(train_pairs):
        # Sao chép ảnh
        dst_img = os.path.join(output_dir, 'train', 'images', os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        
        # Sao chép nhãn
        dst_label = os.path.join(output_dir, 'train', 'labels', os.path.basename(label_path))
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(train_pairs)} training pairs")
    
    print("\nCopying validation files...")
    for i, (img_path, label_path) in enumerate(val_pairs):
        # Sao chép ảnh
        dst_img = os.path.join(output_dir, 'val', 'images', os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        
        # Sao chép nhãn
        dst_label = os.path.join(output_dir, 'val', 'labels', os.path.basename(label_path))
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(val_pairs)} validation pairs")
    
    print(f"\nDataset split completed!")
    print(f"Output directory: {output_dir}")
    print(f"Training set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

def main():
    current_dir = os.getcwd()
    source_dir = r"D:\coding\count-car-project\data\raw\20250723"  # Current directory contains the train folder
    output_dir = r"D:\coding\count-car-project\data\train_data\250723"
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
    print(f"The split dataset is available in: {output_dir}")

if __name__ == "__main__":
    main()
