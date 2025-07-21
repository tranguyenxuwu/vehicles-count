#!/usr/bin/env python3
"""
Test script for adaptive depth heatmap functionality.
This script tests the improved heatmap generation based on object detection density.
"""

import cv2
import os
from ultralytics import YOLO
from generate_heatmap import (
    generate_depth_heatmap_video, 
    generate_depth_heatmap_image,
    estimate_adaptive_depth_map,
    estimate_background_depth_map,
    estimate_depth_from_object_density
)

def test_adaptive_depth_methods():
    """Test different depth estimation methods."""
    print("=== Testing Adaptive Depth Heatmap Methods ===")
    
    # Load model
    model_path = "yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please ensure YOLO model is available.")
        return
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    # Test video path
    video_path = "data/test_videos/test.mp4"
    
    if not os.path.exists(video_path):
        print(f"Test video not found at: {video_path}")
        print("Available test videos:")
        test_video_dir = "data/test_videos/"
        if os.path.exists(test_video_dir):
            for file in os.listdir(test_video_dir):
                if file.endswith(('.mp4', '.avi', '.mov', '.MOV')):
                    print(f"  - {file}")
                    video_path = os.path.join(test_video_dir, file)
                    break
        else:
            print("No test videos directory found!")
            return
    
    if not os.path.exists(video_path):
        print("No suitable test video found!")
        return
        
    print(f"Using test video: {video_path}")
    
    # Test different depth estimation methods
    methods = {
        'perspective': 'Perspective-based depth (original method)',
        'density': 'Object density-based depth (new method)', 
        'combined': 'Adaptive combined depth (recommended)'
    }
    
    print("\n" + "="*60)
    print("TESTING ADAPTIVE DEPTH HEATMAP METHODS")
    print("="*60)
    
    for method, description in methods.items():
        print(f"\n🎯 Testing: {description}")
        print("-" * 50)
        
        try:
            # Generate heatmap video with current method
            output_video = generate_depth_heatmap_video(
                model=model,
                video_path=video_path,
                conf=0.5,
                depth_method=method
            )
            print(f"✅ Success! Generated: {output_video}")
            
        except Exception as e:
            print(f"❌ Error with method '{method}': {str(e)}")
    
    print("\n" + "="*60)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*60)
    print("""
    🔍 PERSPECTIVE METHOD:
    - Uses vertical position in frame to estimate depth
    - Objects lower = closer, objects higher = farther
    - Simple but may not reflect real object distribution
    
    🎯 DENSITY METHOD (NEW):
    - Analyzes object detection density in different regions
    - Areas with more detected objects = considered closer
    - Better reflects actual traffic/object concentration
    
    🚀 COMBINED METHOD (RECOMMENDED):
    - Intelligently combines both approaches
    - Uses density-based method when objects are detected
    - Falls back to perspective method for empty areas
    - Adaptive weighting based on detection count
    
    ✨ Key Benefits:
    - More accurate depth representation in high-traffic areas
    - Better visualization of object concentration zones
    - Improved depth estimation for traffic analysis
    """)

def test_with_image():
    """Test depth heatmap generation with image."""
    print("\n=== Testing with Image ===")
    
    # Load model
    model = YOLO("yolo11n.pt")
    
    # Look for test images
    test_image_dirs = [
        "data/test/images",
        "data/images", 
        "data/raw/images"
    ]
    
    image_path = None
    for img_dir in test_image_dirs:
        if os.path.exists(img_dir):
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(img_dir, image_files[0])
                break
    
    if image_path is None:
        print("No test images found.")
        return
    
    print(f"Using test image: {image_path}")
    
    # Test different methods with image
    methods = ['perspective', 'density', 'combined']
    
    for method in methods:
        try:
            output_image, depth_info = generate_depth_heatmap_image(
                model=model,
                image_path=image_path,
                conf=0.3,
                depth_method=method,
                save_dir_base="test_depth_heatmap"
            )
            print(f"✅ {method.capitalize()} method - Image: {output_image}")
            print(f"   Detected objects: {len(depth_info)}")
            
            # Show depth info for first few objects
            for i, (obj_name, info) in enumerate(list(depth_info.items())[:3]):
                print(f"   {obj_name}: depth={info['depth']:.1f}m, conf={info['confidence']:.2f}")
                if i >= 2:  # Show only first 3 objects
                    break
                    
        except Exception as e:
            print(f"❌ Error with {method} method: {str(e)}")

def main():
    """Main test function."""
    print("🚗 Adaptive Depth Heatmap Test Suite 🚗")
    print("=" * 50)
    
    # Test with image first (faster)
    test_with_image()
    
    # Test with video processing
    test_adaptive_depth_methods()
    
    print("\n✅ Testing completed!")
    print("\nKey improvements:")
    print("• Density-based depth: Areas with more detected objects are considered closer")
    print("• Combined method: Adaptively weights perspective and density methods")
    print("• Better depth estimation in object-rich areas")

if __name__ == "__main__":
    main()
