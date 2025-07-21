import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ultralytics import YOLO
import os
from pathlib import Path
from datetime import datetime
import tempfile

# Try to import scipy, use fallback if not available
try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_gaussian_filter = None

def apply_gaussian_smoothing(input_array, sigma):
    """Apply Gaussian smoothing with fallback if scipy not available."""
    if SCIPY_AVAILABLE and scipy_gaussian_filter is not None:
        return scipy_gaussian_filter(input_array, sigma)
    else:
        # Simple alternative using OpenCV Gaussian blur
        if len(input_array.shape) == 2:
            return cv2.GaussianBlur(input_array.astype(np.float32), (5, 5), sigma)
        return input_array

def estimate_background_depth_map(frame):
    """
    Generate a depth map based on perspective (vertical position in frame).
    Objects lower in the frame are typically closer (assuming ground plane).
    
    Args:
        frame: Input image/frame
    
    Returns:
        2D depth map of the scene based on perspective
    """
    height, width = frame.shape[:2]
    
    # Create a linear gradient from top (far) to bottom (near)
    y_coords = np.arange(height).reshape(-1, 1)
    depth_map = np.broadcast_to(y_coords, (height, width)).astype(np.float32)
    
    # Normalize to 1-10 range (top=10, bottom=1)
    depth_map = 10.0 - (depth_map / height) * 9.0
    
    return depth_map

def estimate_depth_from_object_density(frame, detections, grid_size=32):
    """
    Generate a depth map based on object detection density in different regions.
    Areas with higher object detection density are considered closer.
    
    Args:
        frame: Input image/frame
        detections: List of detection results from YOLO
        grid_size: Size of grid cells for density calculation
    
    Returns:
        2D depth map based on object detection density
    """
    height, width = frame.shape[:2]
    
    # Create grid for density calculation
    grid_h = height // grid_size
    grid_w = width // grid_size
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    # Count detections in each grid cell
    for detection in detections:
        if detection.boxes is not None:
            for box in detection.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Convert to grid coordinates
                grid_x = min(center_x // grid_size, grid_w - 1)
                grid_y = min(center_y // grid_size, grid_h - 1)
                
                # Increase density count
                density_grid[grid_y, grid_x] += 1.0
    
    # Apply Gaussian smoothing to density grid
    smoothed_density = apply_gaussian_smoothing(density_grid, 1.0)
    
    # Resize density grid to match frame size
    density_map = cv2.resize(smoothed_density, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize density to depth values
    if np.max(density_map) > 0:
        # Higher density = closer (lower depth value)
        # Invert so high density areas have low depth values (close)
        normalized_density = density_map / np.max(density_map)
        depth_map = 10.0 - (normalized_density * 9.0)  # Range: 1-10 (close-far)
    else:
        # No detections, use perspective-based depth
        depth_map = estimate_background_depth_map(frame)
    
    # Combine with perspective depth for areas with no detections
    perspective_depth = estimate_background_depth_map(frame)
    
    # Weight combination: 70% density-based, 30% perspective-based
    alpha = 0.7
    combined_depth = alpha * depth_map + (1 - alpha) * perspective_depth
    
    return combined_depth

def estimate_adaptive_depth_map(frame, detections, method='combined'):
    """
    Generate an adaptive depth map using multiple methods.
    
    Args:
        frame: Input image/frame
        detections: List of detection results from YOLO
        method: 'perspective', 'density', or 'combined'
    
    Returns:
        2D depth map of the scene
    """
    if method == 'perspective':
        return estimate_background_depth_map(frame)
    elif method == 'density':
        return estimate_depth_from_object_density(frame, detections)
    elif method == 'combined':
        # Combine perspective and density-based methods
        perspective_depth = estimate_background_depth_map(frame)
        
        # Check if we have detections
        has_detections = any(detection.boxes is not None and len(detection.boxes) > 0 
                           for detection in detections)
        
        if has_detections:
            density_depth = estimate_depth_from_object_density(frame, detections)
            # Adaptive weighting based on number of detections
            total_detections = sum(len(detection.boxes) if detection.boxes is not None else 0 
                                 for detection in detections)
            
            # More detections = more weight to density method
            density_weight = min(0.8, total_detections * 0.1)
            perspective_weight = 1.0 - density_weight
            
            combined_depth = density_weight * density_depth + perspective_weight * perspective_depth
            return combined_depth
        else:
            return perspective_depth
    else:
        raise ValueError("Method must be 'perspective', 'density', or 'combined'")

def estimate_depth_from_bbox_size(bbox, image_height, focal_length_estimate=800):
    """
    Estimate depth based on bounding box size.
    Assumes larger objects are closer, smaller objects are farther.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_height: Height of the image
        focal_length_estimate: Estimated focal length of camera
    
    Returns:
        Estimated depth/distance value
    """
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    bbox_area = bbox_height * bbox_width
    
    # Normalize by image size
    normalized_area = bbox_area / (image_height * image_height)
    
    # Simple inverse relationship: smaller area = farther distance
    # This is a simplified model - in reality, you'd need camera calibration
    estimated_distance = 1.0 / (normalized_area + 0.001)  # Add small epsilon to avoid division by zero
    
    return min(estimated_distance, 100.0)  # Cap maximum distance

def estimate_depth_from_position(bbox, image_height, image_width):
    """
    Estimate depth based on object position in frame.
    Objects lower in the frame are typically closer (assuming ground plane).
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_height: Height of the image
        image_width: Width of the image
    
    Returns:
        Estimated depth based on vertical position
    """
    x1, y1, x2, y2 = bbox
    center_y = (y1 + y2) / 2
    
    # Normalize position (0 = top, 1 = bottom)
    normalized_y = center_y / image_height
    
    # Objects lower in frame are closer (inverse relationship)
    # Scale from 1 (close) to 10 (far)
    depth_estimate = 10.0 - (normalized_y * 9.0)
    
    return max(depth_estimate, 1.0)

def create_depth_colormap(depth_values, colormap='plasma'):
    """
    Create a color mapping for depth values.
    
    Args:
        depth_values: List of depth/distance values
        colormap: Matplotlib colormap name
    
    Returns:
        Color mapping function
    """
    if not depth_values:
        return lambda x: (255, 255, 255)
    
    min_depth = min(depth_values)
    max_depth = max(depth_values)
    depth_range = max_depth - min_depth
    
    if depth_range == 0:
        return lambda x: (255, 255, 255)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    def map_depth_to_color(depth):
        # Normalize depth to 0-1 range
        normalized = (depth - min_depth) / depth_range
        # Get color from colormap
        rgba = cmap(normalized)
        # Convert to BGR for OpenCV
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        return bgr
    
    return map_depth_to_color

def create_depth_heatmap_overlay(frame, depth_map, alpha=0.6):
    """
    Create a heatmap overlay on the frame showing depth information.
    
    Args:
        frame: Input video frame
        depth_map: 2D array of depth values
        alpha: Transparency of the overlay
    
    Returns:
        Frame with heatmap overlay
    """
    # Normalize depth map to 0-255 range
    if np.max(depth_map) > np.min(depth_map):
        normalized_depth = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
    else:
        normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    
    # Blend with original frame
    blended = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
    
    return blended

def generate_depth_heatmap_video(model, video_path, output_path=None, conf=0.5, 
                                save_dir_base="depth_heatmap", depth_method='combined'):
    """
    Generate a video with adaptive depth heatmap overlay.
    
    Args:
        model: YOLO model for object detection
        video_path: Path to input video
        output_path: Path for output video (optional)
        conf: Confidence threshold for detections
        save_dir_base: Base directory for saving output
        depth_method: 'perspective', 'density', or 'combined'
    
    Returns:
        Path to output video
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    if output_path is None:
        method_suffix = f"_{depth_method}" if depth_method != 'combined' else ""
        output_path = os.path.join(save_dir, f"adaptive_depth_heatmap{method_suffix}_{Path(video_path).stem}.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"Processing {total_frames} frames with {depth_method} depth estimation...")
    
    # Fixed depth range for depth estimation
    min_depth, max_depth = 1.0, 10.0
    
    # Create color mapper
    color_mapper = create_depth_colormap([min_depth, max_depth])
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run object detection first to get detections for adaptive depth
        results = model(frame, conf=conf, verbose=False)
        
        # Generate adaptive depth map using selected method
        depth_map = estimate_adaptive_depth_map(frame, results, method=depth_method)
        
        # Create heatmap overlay
        heatmap_frame = create_depth_heatmap_overlay(frame, depth_map, alpha=0.6)
        
        # Overlay object detection results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Get depth value from the depth map at object center
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                    center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                    object_depth = depth_map[center_y, center_x]
                    
                    # Draw bounding box with white color for visibility
                    cv2.rectangle(heatmap_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    
                    # Add label with adaptive depth at object location
                    label = f"{class_name}: {confidence:.2f} | D: {object_depth:.1f}m"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(heatmap_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), (0, 0, 0), -1)
                    cv2.putText(heatmap_frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add depth scale legend
        legend_height = 200
        legend_width = 30
        legend_x = width - legend_width - 20
        legend_y = 20
        
        # Create depth scale
        for i in range(legend_height):
            depth_val = min_depth + (max_depth - min_depth) * (1 - i / legend_height)
            color = color_mapper(depth_val)
            cv2.rectangle(heatmap_frame, 
                         (legend_x, legend_y + i), 
                         (legend_x + legend_width, legend_y + i + 1), 
                         color, -1)
        
        # Add scale labels
        cv2.putText(heatmap_frame, f"{max_depth:.1f}m", 
                   (legend_x + legend_width + 5, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(heatmap_frame, f"{min_depth:.1f}m", 
                   (legend_x + legend_width + 5, legend_y + legend_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display method name
        method_display = {
            'perspective': 'Perspective Depth',
            'density': 'Density-based Depth',
            'combined': 'Adaptive Depth'
        }
        cv2.putText(heatmap_frame, method_display.get(depth_method, 'Adaptive Depth'), 
                   (legend_x - 60, legend_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info with object count from current results
        object_count = sum(len(result.boxes) if result.boxes is not None else 0 for result in results)
        info_text = f"Frame: {frame_count + 1}/{total_frames} | Objects: {object_count} | Method: {depth_method.title()}"
        cv2.putText(heatmap_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(heatmap_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"Adaptive depth heatmap video saved to: {output_path}")
    return output_path

def generate_depth_heatmap_image(model, image_path, output_path=None, conf=0.5, 
                                save_dir_base="depth_heatmap", depth_method='combined'):
    """
    Generate an image with adaptive depth heatmap overlay.
    
    Args:
        model: YOLO model for object detection
        image_path: Path to input image
        output_path: Path for output image (optional)
        conf: Confidence threshold for detections
        save_dir_base: Base directory for saving output
        depth_method: 'perspective', 'density', or 'combined'
    
    Returns:
        Path to output image, depth information
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    if output_path is None:
        method_suffix = f"_{depth_method}" if depth_method != 'combined' else ""
        output_path = os.path.join(save_dir, f"adaptive_depth_heatmap{method_suffix}_{Path(image_path).stem}.jpg")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    height, width = image.shape[:2]
    
    # Run inference first to get detections
    results = model(image, conf=conf, verbose=False)
    
    depth_info = {}
    
    # Generate adaptive depth map using selected method
    depth_map = estimate_adaptive_depth_map(image, results, method=depth_method)
    
    # Create heatmap overlay
    heatmap_image = create_depth_heatmap_overlay(image, depth_map, alpha=0.6)
    
    # Overlay object detections
    for result in results:
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[cls_id]
                
                # Get depth value from the depth map at object center
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))
                object_depth = depth_map[center_y, center_x]
                
                # Store depth information
                depth_info[f"{class_name}_{i+1}"] = {
                    'depth': float(object_depth),
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'type': f'adaptive_{depth_method}_depth'
                }
                
                # Draw bounding box with white color for visibility
                cv2.rectangle(heatmap_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                
                # Add label with adaptive depth at object location
                label = f"{class_name}: {confidence:.2f} | D: {object_depth:.1f}m"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(heatmap_image, (int(x1), int(y1) - label_size[1] - 10), 
                            (int(x1) + label_size[0], int(y1)), (0, 0, 0), -1)
                cv2.putText(heatmap_image, label, (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Set fixed depth range for depth estimation
    min_depth, max_depth = 1.0, 10.0
    
    # Add depth scale legend
    legend_height = 200
    legend_width = 30
    legend_x = width - legend_width - 20
    legend_y = 20
    
    # Create color mapper for legend
    color_mapper = create_depth_colormap([min_depth, max_depth])
    
    # Create depth scale
    for i in range(legend_height):
        depth_val = min_depth + (max_depth - min_depth) * (1 - i / legend_height)
        color = color_mapper(depth_val)
        cv2.rectangle(heatmap_image, 
                     (legend_x, legend_y + i), 
                     (legend_x + legend_width, legend_y + i + 1), 
                     color, -1)
    
    # Add scale labels
    cv2.putText(heatmap_image, f"{max_depth:.1f}m", 
               (legend_x + legend_width + 5, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(heatmap_image, f"{min_depth:.1f}m", 
               (legend_x + legend_width + 5, legend_y + legend_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display method name
    method_display = {
        'perspective': 'Perspective Depth',
        'density': 'Density-based Depth', 
        'combined': 'Adaptive Depth'
    }
    cv2.putText(heatmap_image, method_display.get(depth_method, 'Adaptive Depth'), 
               (legend_x - 60, legend_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save result
    cv2.imwrite(output_path, heatmap_image)
    
    print(f"Adaptive depth heatmap image saved to: {output_path}")
    return output_path, depth_info

# Example usage
if __name__ == "__main__":
    # Load model
    model = YOLO("yolo11n.pt")  # or your trained model
    
    # Example for video with different depth methods
    video_path = "data/test_videos/test.mp4"
    if os.path.exists(video_path):
        # Test different methods
        methods = ['perspective', 'density', 'combined']
        
        for method in methods:
            print(f"Generating {method} depth heatmap video...")
            output_video = generate_depth_heatmap_video(
                model, 
                video_path, 
                conf=0.5,
                depth_method=method
            )
            print(f"Generated {method} depth heatmap video: {output_video}")
    
    # Example for image with adaptive depth
    image_path = "data/test/images/sample.jpg"  # Update with actual image path
    if os.path.exists(image_path):
        print("Generating adaptive depth heatmap image...")
        output_image, depth_data = generate_depth_heatmap_image(
            model, 
            image_path, 
            conf=0.5,
            depth_method='combined'
        )
        print(f"Generated adaptive depth heatmap image: {output_image}")
        print("Adaptive depth information:", depth_data)
