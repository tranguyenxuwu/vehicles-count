import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
from datetime import datetime

def draw_obb_and_count(image, results, class_names, colors):
    """Draw oriented bounding boxes and count objects"""
    object_counts = {}
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[cls_id]
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                color = colors.get(class_name, (255, 255, 255))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), color, -1)
                cv2.putText(image, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if hasattr(result, 'obb') and result.obb is not None:
            obb_boxes = result.obb
            for obb in obb_boxes:
                points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                cv2.polylines(image, [points], True, (255, 0, 0), 2)
                cls_id = int(obb.cls[0])
                confidence = float(obb.conf[0])
                class_name = class_names[cls_id]
                label = f"OBB {class_name}: {confidence:.2f}"
                cv2.putText(image, label, (points[0][0], points[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image, object_counts

def display_counts(image, object_counts):
    """Display object counts on image"""
    y_offset = 30
    total_objects = sum(object_counts.values())
    cv2.putText(image, f"Total Objects: {total_objects}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    return image

def predict_image(model, source_path, save_dir_base="predict", conf=0.5):
    """Predict and visualize with OBB and counts for image. Trả về ảnh kết quả, object_counts, và đường dẫn file đã lưu."""
    
    # Create a timestamped directory for the current prediction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    results = model.predict(
        source=source_path,
        save=False,
        conf=conf,
        device=model.device
    )
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    image_with_boxes, object_counts = draw_obb_and_count(image, results, class_names, colors)
    final_image = display_counts(image_with_boxes, object_counts)
    
    output_path = os.path.join(save_dir, f"result_{Path(source_path).stem}.jpg")
    cv2.imwrite(output_path, final_image)
    return final_image, object_counts, output_path

def predict_video(model, video_path, save_dir_base="predict", conf=0.6, progress_callback=None):
    """Predict on video with object counting and tracking. Trả về đường dẫn video đã lưu."""
    
    # Create a timestamped directory for the current prediction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = os.path.join(save_dir, f"result_{Path(video_path).stem}.mp4")
    fourcc = 0x31637661  # 'avc1' codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    tracked_objects = {}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, conf=conf, device=model.device, verbose=False)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = class_names[cls]
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[cls]
                color = colors.get(class_name, (255, 255, 255))
                label = f"ID:{track_id} {class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        final_frame = display_counts(frame, {cls: list(tracked_objects.values()).count(cls) for cls in set(tracked_objects.values())})
        out.write(final_frame)
        frame_count += 1
        
        # Update progress if callback provided
        if progress_callback and total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_callback(progress, frame_count, total_frames)
            
    cap.release()
    out.release()
    return output_path