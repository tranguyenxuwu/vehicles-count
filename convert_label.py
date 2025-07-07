"""
Polygon to Bounding Box Converter
=================================

This script converts polygon annotations to bounding box format for YOLO training.

Input format:  class_id x1 y1 x2 y2 x3 y3 ... xn yn (polygon vertices)
Output format: class_id x_center y_center width height (bounding box)

Usage:
1. Set INPUT_DIR to the directory containing polygon annotation files (.txt)
2. Set OUTPUT_DIR to the directory where bounding box annotations will be saved
3. Run the script: python convert_label.py

All coordinates are assumed to be normalized (0-1 range).
"""

import os
import glob
from pathlib import Path

def polygon_to_bbox(polygon_coords):
    """
    Convert polygon coordinates to bounding box format.
    
    Args:
        polygon_coords: List of coordinates [x1, y1, x2, y2, ..., xn, yn]
    
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized format
    """
    # Extract x and y coordinates
    x_coords = [polygon_coords[i] for i in range(0, len(polygon_coords), 2)]
    y_coords = [polygon_coords[i] for i in range(1, len(polygon_coords), 2)]
    
    # Calculate bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Calculate center point and dimensions
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    return x_center, y_center, width, height

def convert_annotation_file(input_file, output_file):
    """
    Convert a single annotation file from polygon to bounding box format.
    
    Args:
        input_file: Path to input annotation file
        output_file: Path to output annotation file
    """
    converted_lines = []
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the line
                parts = line.split()
                if len(parts) < 3:  # Need at least class_id and one coordinate pair
                    continue
                
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:]]
                
                # Check if we have pairs of coordinates
                if len(coordinates) % 2 != 0:
                    print(f"Warning: Odd number of coordinates in line: {line}")
                    continue
                
                # Convert polygon to bounding box
                x_center, y_center, width, height = polygon_to_bbox(coordinates)
                
                # Format as YOLO bounding box: class_id x_center y_center width height
                converted_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                converted_lines.append(converted_line)
        
        # Write converted annotations
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines))
            if converted_lines:  # Add final newline if there are lines
                f.write('\n')
        
        print(f"Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def convert_annotations(input_dir, output_dir):
    """
    Convert all annotation files in input directory to bounding box format.
    
    Args:
        input_dir: Directory containing polygon annotation files
        output_dir: Directory to save converted bounding box annotations
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in input directory
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} annotation files to convert")
    
    # Process each file
    for input_file in txt_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        convert_annotation_file(input_file, output_file)
    
    print(f"\nConversion completed! {len(txt_files)} files processed.")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    # Configuration - modify these paths as needed
    INPUT_DIR = "check-label"  # Directory containing polygon annotations
    OUTPUT_DIR = "bbox-labels"  # Directory to save bounding box annotations
    
    # You can also use absolute paths:
    # INPUT_DIR = r"C:\path\to\your\polygon\annotations"
    # OUTPUT_DIR = r"C:\path\to\your\bbox\annotations"
    
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle both relative and absolute paths
    if os.path.isabs(INPUT_DIR):
        input_path = INPUT_DIR
    else:
        input_path = os.path.join(script_dir, INPUT_DIR)
    
    if os.path.isabs(OUTPUT_DIR):
        output_path = OUTPUT_DIR
    else:
        output_path = os.path.join(script_dir, OUTPUT_DIR)
    
    print("Polygon to Bounding Box Converter")
    print("=" * 40)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input directory '{input_path}' does not exist!")
        print("Please modify the INPUT_DIR variable in the script.")
        exit(1)
    
    # Start conversion
    convert_annotations(input_path, output_path)