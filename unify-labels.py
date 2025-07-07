#!/usr/bin/env python3
"""
Script to convert class 5 to class 4 in YOLO format label files.
This script reads all .txt files in the current directory and subdirectories,
finds lines starting with class "5", and converts them to class "4" while
keeping all other information intact.
"""

import os
import glob

def convert_class_5_to_4(file_path):
    """
    Convert class 5 to class 4 in a single file.
    
    Args:
        file_path (str): Path to the file to process
    
    Returns:
        bool: True if any changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            # Check if line starts with "5 " (class 5)
            if line.strip().startswith('5 '):
                # Replace "5 " with "4 " at the beginning of the line
                new_line = '4 ' + line.strip()[2:]
                new_lines.append(new_line + '\n')
                modified = True
                print(f"  Converted: {line.strip()} -> {new_line}")
            else:
                new_lines.append(line)
        
        # Write back to file only if changes were made
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def main():
    """Main function to process all txt files."""
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Processing files in: {current_dir}")
    
    # Find all .txt files recursively
    txt_files = glob.glob(os.path.join(current_dir, '**', '*.txt'), recursive=True)
    
    if not txt_files:
        print("No .txt files found in the current directory and subdirectories.")
        return
    
    print(f"Found {len(txt_files)} .txt files to process.\n")
    
    total_modified = 0
    
    # Process each file
    for file_path in txt_files:
        print(f"Processing: {os.path.relpath(file_path, current_dir)}")
        
        if convert_class_5_to_4(file_path):
            total_modified += 1
            print(f"  ✓ Modified")
        else:
            print(f"  - No changes needed")
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {len(txt_files)}")
    print(f"Files modified: {total_modified}")
    
    if total_modified == 0:
        print("No files contained class 5 labels.")

if __name__ == "__main__":
    main()