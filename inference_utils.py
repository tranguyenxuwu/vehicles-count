import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
from datetime import datetime

def draw_obb_and_count(image, results, class_names, colors):
    """Vẽ khung bao và đếm đối tượng"""
    # Khởi tạo từ điển để lưu số lượng từng loại đối tượng được phát hiện
    object_counts = {}
    
    # Duyệt qua tất cả kết quả phát hiện
    for result in results:
        # Lấy các khung bao từ kết quả hiện tại
        boxes = result.boxes
        if boxes is not None:
            # Xử lý từng khung bao được phát hiện
            for box in boxes:
                # Trích xuất ID lớp, độ tin cậy và tên lớp
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[cls_id]
                
                # Khởi tạo hoặc tăng số đếm cho loại đối tượng này
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                
                # Trích xuất tọa độ khung bao (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Lấy màu cho lớp này hoặc sử dụng màu trắng mặc định
                color = colors.get(class_name, (255, 255, 255))
                
                # Vẽ khung chữ nhật bao quanh đối tượng được phát hiện
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Tạo nhãn với tên lớp và độ tin cậy (chuyển sang phần trăm)
                label = f"{class_name}: {confidence*100:.0f}%"
                
                # Tính kích thước văn bản cho nền nhãn
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Vẽ hình chữ nhật làm nền cho nhãn
                cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), color, -1)
                
                # Vẽ nhãn lên trên nền
                cv2.putText(image, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Kiểm tra nếu kết quả có khung
        if hasattr(result, 'obb') and result.obb is not None:
            obb_boxes = result.obb
            # Xử lý từng khung bao định hướng
            for obb in obb_boxes:
                # Lấy 8 điểm góc của khung bao định hướng (4 góc, mỗi góc có x,y)
                points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                
                # Vẽ khung bao định hướng bằng các đường nối
                cv2.polylines(image, [points], True, (255, 0, 0), 2)
                
                # Trích xuất thông tin lớp cho OBB
                cls_id = int(obb.cls[0])
                confidence = float(obb.conf[0])
                class_name = class_names[cls_id]
                
                # Tạo nhãn cho OBB (chuyển sang phần trăm)
                label = f"OBB {class_name}: {confidence*100:.0f}%"
                
                # Vẽ nhãn OBB tại điểm góc đầu tiên
                cv2.putText(image, label, (points[0][0], points[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image, object_counts

def display_counts(image, object_counts):
    """Hiển thị số lượng đối tượng trên ảnh"""
    # Vị trí bắt đầu hiển thị (góc trên bên trái)
    y_offset = 30
    
    # Tính và hiển thị tổng số đối tượng được phát hiện
    total_objects = sum(object_counts.values())
    cv2.putText(image, f"Total Objects: {total_objects}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Chuyển xuống dòng tiếp theo
    y_offset += 30
    
    # Hiển thị số lượng cho từng loại đối tượng
    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Chuyển xuống dòng cho lớp tiếp theo
        y_offset += 25
    
    return image

def predict_image(model, source_path, save_dir_base="predict", conf=0.5):
    """Dự đoán và hiển thị OBB và số lượng cho ảnh"""
    
    # Tạo thư mục có timestamp duy nhất cho phiên dự đoán này
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Chạy dự đoán mô hình trên ảnh đầu vào
    results = model.predict(
        source=source_path,
        save=False,  # Không tự động lưu, ta sẽ lưu thủ công với chú thích
        conf=conf,   # Ngưỡng tin cậy cho phát hiện
        device=model.device
    )
    
    # Tải ảnh gốc bằng OpenCV
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")
    
    # Lấy tên lớp từ mô hình và tạo màu ngẫu nhiên cho mỗi lớp
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    
    # Vẽ khung bao và đếm đối tượng
    image_with_boxes, object_counts = draw_obb_and_count(image, results, class_names, colors)
    
    # Thêm lớp hiển thị số đếm lên ảnh
    final_image = display_counts(image_with_boxes, object_counts)
    
    # Lưu ảnh kết quả có chú thích
    output_path = os.path.join(save_dir, f"result_{Path(source_path).stem}.jpg")
    cv2.imwrite(output_path, final_image)
    
    return final_image, object_counts, output_path

def predict_video(model, video_path, save_dir_base="predict", conf=0.6, progress_callback=None):
    """Dự đoán video với đếm và theo dõi đối tượng"""
    
    # Tạo thư mục có timestamp duy nhất cho dự đoán video này
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") # dạng : 20250722_1530
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Mở video đầu vào và lấy thuộc tính
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))           # Số khung hình/giây
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Chiều rộng video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Chiều cao video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Tổng số khung hình
    
    # Thiết lập writer video đầu ra
    output_path = os.path.join(save_dir, f"result_{Path(video_path).stem}.mp4")
    fourcc = 0x31637661  # Codec 'avc1' cho định dạng MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Khởi tạo biến theo dõi
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    tracked_objects = {}  # Từ điển lưu track_id -> class_name
    frame_count = 0
    
    # Xử lý video từng khung hình
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Hết video
            
        # Chạy theo dõi trên khung hình hiện tại (duy trì ID đối tượng qua các khung)
        results = model.track(frame, persist=True, conf=conf, device=model.device, verbose=False)
        
        # Xử lý kết quả theo dõi nếu có đối tượng được phát hiện
        if results[0].boxes.id is not None:
            # Trích xuất thông tin theo dõi
            boxes = results[0].boxes.xyxy.cpu()          # Tọa độ khung bao
            track_ids = results[0].boxes.id.int().cpu().tolist()  # ID theo dõi duy nhất
            clss = results[0].boxes.cls.cpu().tolist()   # ID lớp
            
            # Xử lý từng đối tượng được theo dõi
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Ghi lại lớp của đối tượng này trong từ điển theo dõi
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = class_names[cls]
                
                # Vẽ khung bao và nhãn cho đối tượng được theo dõi
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[cls]
                color = colors.get(class_name, (255, 255, 255))
                
                # Nhãn bao gồm cả ID theo dõi và tên lớp
                label = f"ID:{track_id} {class_name}"
                
                # Vẽ hình chữ nhật và nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Đếm đối tượng duy nhất theo lớp (dựa trên tất cả đối tượng đã theo dõi)
        class_counts = {cls: list(tracked_objects.values()).count(cls) for cls in set(tracked_objects.values())}
        
        # Thêm hiển thị số đếm vào khung hình
        final_frame = display_counts(frame, class_counts)
        
        # Ghi khung hình đã xử lý vào video đầu ra
        out.write(final_frame)
        frame_count += 1
        
        # Cập nhật tiến độ nếu có hàm callback
        if progress_callback and total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)  # Đảm bảo tiến độ không vượt quá 100%
            progress_callback(progress, frame_count, total_frames)
            
    # Dọn dẹp tài nguyên video
    cap.release()
    out.release()
    
    return output_path