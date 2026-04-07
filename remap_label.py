#!/usr/bin/env python3
"""
Remap VisDrone labels: merge class 1 (people) into class 0 (pedestrian).
After remapping: 0=pedestrian, 1=car, 2=truck, 3=bus, 4=motor
"""

import os
from pathlib import Path


def remap_labels(label_file: Path, output_file: Path) -> None:
    """
    Remap label classes:
    - Class 0 (pedestrian) stays 0
    - Class 1 (people) -> 0 (pedestrian)
    - Class 2 (car) -> 1
    - Class 3 (truck) -> 2
    - Class 4 (van) -> 1 (car) - already merged
    - Class 5 (bus) -> 3
    - Class 6 (motor) -> 4
    
    Old: ['pedestrian', 'people', 'car', 'truck', 'van', 'bus', 'motor']
    New: ['pedestrian', 'car', 'truck', 'bus', 'motor']
    """
    # Mapping from old class to new class
    class_mapping = {
        0: 0,  # pedestrian -> pedestrian
        1: 0,  # people -> pedestrian (merge)
        2: 1,  # car -> car
        3: 2,  # truck -> truck
        4: 1,  # van -> car (merge)
        5: 3,  # bus -> bus
        6: 4,  # motor -> motor
    }
    
    if not label_file.exists():
        return
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        old_class = int(parts[0])
        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            remapped_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(remapped_lines)


def process_directory_inplace(label_dir: Path) -> int:
    """Process all label files in a directory in-place."""
    if not label_dir.exists():
        print(f"Directory not found: {label_dir}")
        return 0
    
    count = 0
    for label_file in label_dir.glob("*.txt"):
        remap_labels(label_file, label_file)
        count += 1
    
    return count


def main():
    base_path = Path("/home/vscode/iDragonCloud/ComputerVision/visdrone")
    labels_base = base_path / "labels"
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        label_dir = labels_base / split
        
        if label_dir.exists():
            count = process_directory_inplace(label_dir)
            print(f"Processed {count} files in {split}/")
        else:
            print(f"Skipping {split}/ - directory does not exist")
    
    print("\nRemapping complete!")
    print("New class mapping:")
    print("  0: pedestrian (merged with people)")
    print("  1: car (merged with van)")
    print("  2: truck")
    print("  3: bus")
    print("  4: motor")


if __name__ == "__main__":
    main()
