import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os
import argparse

def draw_obb_and_count(image, results, class_names, colors):
    """Draw oriented bounding boxes and count objects"""
    object_counts = {}
    
    for result in results:
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                # Get class info
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[cls_id]
                
                # Count objects
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get color for class
                color = colors.get(class_name, (255, 255, 255))  # Default to white
                
                # Draw regular bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), color, -1)
                cv2.putText(image, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Check for OBB (Oriented Bounding Boxes)
        if hasattr(result, 'obb') and result.obb is not None:
            obb_boxes = result.obb
            for obb in obb_boxes:
                # Get OBB coordinates (4 corner points)
                points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                
                # Draw OBB
                cv2.polylines(image, [points], True, (255, 0, 0), 2)
                
                # Get class info for OBB
                cls_id = int(obb.cls[0])
                confidence = float(obb.conf[0])
                class_name = class_names[cls_id]
                
                # Draw OBB label
                label = f"OBB {class_name}: {confidence:.2f}"
                cv2.putText(image, label, (points[0][0], points[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image, object_counts

def display_counts(image, object_counts):
    """Display object counts on image"""
    y_offset = 30
    total_objects = sum(object_counts.values())
    
    # Display total count
    cv2.putText(image, f"Total Objects: {total_objects}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    # Display count for each class
    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    return image

def predict_with_visualization(model, source_path, save_dir="runs/predict"):
    """Predict and visualize with OBB and counts"""
    # Make prediction
    results = model.predict(
        source=source_path,
        save=False,  # We'll save manually after drawing
        conf=0.5,
        device=model.device
    )
    
    # Load image
    image = cv2.imread(source_path)
    if image is None:
        print(f"Could not load image: {source_path}")
        return
    
    # Get class names
    class_names = model.names
    
    # Generate colors for each class
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    
    # Draw OBB and count objects
    image_with_boxes, object_counts = draw_obb_and_count(image, results, class_names, colors)
    
    # Display counts on image
    final_image = display_counts(image_with_boxes, object_counts)
    
    # Print counts to console
    print("\nObject Detection Results:")
    print("-" * 30)
    total_objects = sum(object_counts.values())
    print(f"Total Objects Detected: {total_objects}")
    
    for class_name, count in object_counts.items():
        print(f"{class_name}: {count}")
    
    # Save result
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"result_{Path(source_path).stem}.jpg")
    cv2.imwrite(output_path, final_image)
    print(f"\nResult saved to: {output_path}")
    
    # Display image (optional)
    cv2.imshow("Detection Results", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return results, object_counts

def predict_video_with_counts(model, video_path, save_dir="runs/predict"):
    """Predict on video with object counting and tracking"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"result_{Path(video_path).stem}.mp4")
    fourcc = 0x7634706d  # Equivalent to 'mp4v'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    class_names = model.names
    
    # Generate colors for each class
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    
    tracked_objects = {}
    next_object_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict and track on frame
        results = model.track(frame, persist=True, conf=0.5, device=model.device, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = class_names[cls]

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[cls]
                color = colors.get(class_name, (255, 255, 255))
                label = f"ID:{track_id} {class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display total counts on the frame
        final_frame = display_counts(frame, {cls: list(tracked_objects.values()).count(cls) for cls in set(tracked_objects.values())})
        
        # Write frame
        out.write(final_frame)
        frame_count += 1
        
        if frame_count % 10 == 0:  # Print every 10 frames
            total_objects = len(tracked_objects)
            print(f"Frame {frame_count}: {total_objects} unique objects tracked so far")
    
    cap.release()
    out.release()
    print(f"Video processing completed. Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection")
    parser.add_argument("--image", type=str, help="Path to the image file for prediction.")
    parser.add_argument("--video", type=str, help="Path to the video file for prediction.")
    args = parser.parse_args()

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load your trained model
    model_path = "./runs/detect/train7/weights/best.pt"
    # if not os.path.exists(model_path):
    #     print(f"Model not found at {model_path}")
    #     print("Using pretrained model instead...")
    #     model_path = "yolo11n.pt"
    
    # Load the model
    model = YOLO(model_path)
    model.to(device)
    
    # Example usage
    if args.image:
        if os.path.exists(args.image):
            print(f"Processing image: {args.image}")
            predict_with_visualization(model, args.image)
        else:
            print(f"Image not found: {args.image}")
    
    # Example for video
    if args.video:
        if os.path.exists(args.video):
            print(f"Processing video: {args.video}")
            predict_video_with_counts(model, args.video)
        else:
            print(f"Video not found: {args.video}")

if __name__ == '__main__':
    main()