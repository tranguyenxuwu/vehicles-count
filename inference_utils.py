import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
from datetime import datetime

# --- SAHI imports ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def get_sahi_model(model_path, conf=0.5, device="mps"):
    """Tạo SAHI AutoDetectionModel từ đường dẫn model YOLO.
    
    Args:
        model_path: Đường dẫn tới file model YOLO (.pt)
        conf: Ngưỡng tin cậy
        device: Thiết bị chạy inference ('cpu' hoặc 'cuda:0')
    
    Returns:
        AutoDetectionModel đã khởi tạo
    """
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )


def sahi_result_to_boxes(sahi_result, class_names):
    """Chuyển đổi kết quả SAHI thành danh sách các box thống nhất.
    
    Args:
        sahi_result: Kết quả từ get_sliced_prediction()
        class_names: Dict {id: name} hoặc list tên lớp từ model
    
    Returns:
        list[dict] – mỗi phần tử có keys: 'xyxy', 'conf', 'class_name'
    """
    boxes_out = []
    for pred in sahi_result.object_prediction_list:
        bbox = pred.bbox  # BoundingBox object
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        conf = pred.score.value
        cls_id = pred.category.id
        # class_names có thể là dict hoặc list
        if isinstance(class_names, dict):
            name = class_names.get(cls_id, str(cls_id))
        else:
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        boxes_out.append({
            "xyxy": (x1, y1, x2, y2),
            "conf": conf,
            "class_name": name,
        })
    return boxes_out


def draw_boxes_and_count(image, boxes_list, colors, show_label=False):
    """Vẽ khung bao và đếm đối tượng từ danh sách box thống nhất.
    
    Args:
        image: Ảnh numpy BGR
        boxes_list: list[dict] – output của sahi_result_to_boxes()
        colors: Dict {class_name: (B, G, R)}
        show_label: Hiển thị nhãn phần trăm trên mỗi box (mặc định: False)
    
    Returns:
        (image, object_counts)
    """
    object_counts = {}
    for box_info in boxes_list:
        x1, y1, x2, y2 = box_info["xyxy"]
        conf = box_info["conf"]
        class_name = box_info["class_name"]

        object_counts[class_name] = object_counts.get(class_name, 0) + 1

        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if show_label:
            label = f"{class_name}: {conf*100:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(image, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image, object_counts


def draw_obb_and_count(image, results, class_names, colors, show_label=False):
    """Vẽ khung bao và đếm đối tượng (inference gốc – không dùng SAHI)"""
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
                
                if show_label:
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
                
                if show_label:
                    # Tạo nhãn cho OBB (chuyển sang phần trăm)
                    label = f"OBB {class_name}: {confidence*100:.0f}%"
                    
                    # Vẽ nhãn OBB tại điểm góc đầu tiên
                    cv2.putText(image, label, (points[0][0], points[0][1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image, object_counts

def display_counts(image, object_counts, colors=None):
    """Hiển thị số lượng đối tượng trên ảnh với màu tương ứng.
    
    Args:
        image: Ảnh numpy BGR
        object_counts: Dict {class_name: count}
        colors: Dict {class_name: (B, G, R)} – nếu None thì dùng trắng
    """
    # Vị trí bắt đầu hiển thị (góc trên bên trái)
    y_offset = 30
    
    # Tính và hiển thị tổng số đối tượng được phát hiện
    total_objects = sum(object_counts.values())
    cv2.putText(image, f"Total Objects: {total_objects}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Chuyển xuống dòng tiếp theo
    y_offset += 30
    
    # Hiển thị số lượng cho từng loại đối tượng với màu tương ứng
    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"
        color = (255, 255, 255)  # mặc định trắng
        if colors and class_name in colors:
            color = colors[class_name]
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Chuyển xuống dòng cho lớp tiếp theo
        y_offset += 25
    
    return image

def predict_image(model, source_path, save_dir_base="outputs", conf=0.5, iou_threshold=0.45,
                  use_sahi=True, slice_height=512, slice_width=512,
                  overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                  show_label=False):
    """Dự đoán và hiển thị OBB và số lượng cho ảnh.
    
    Khi use_sahi=True (mặc định), sử dụng SAHI tiled inference để cải thiện
    phát hiện đối tượng nhỏ trong ảnh độ phân giải cao.
    
    Args:
        save_dir_base: Thư mục gốc lưu kết quả (mặc định: 'outputs')
        show_label: Hiển thị nhãn phần trăm trên mỗi box (mặc định: False)
    """
    
    # Tạo thư mục outputs/images
    save_dir = os.path.join(save_dir_base, "images")
    os.makedirs(save_dir, exist_ok=True)

    # Tải ảnh gốc bằng OpenCV
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")

    # Lấy tên lớp từ mô hình và tạo màu ngẫu nhiên cho mỗi lớp
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}

    if use_sahi:
        # --- SAHI tiled inference ---
        device = str(model.device)
        model_path = model.ckpt_path  # đường dẫn gốc tới file .pt
        sahi_model = get_sahi_model(model_path, conf=conf, device=device)

        sahi_result = get_sliced_prediction(
            source_path,
            sahi_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=iou_threshold,
        )

        # Chuyển đổi kết quả SAHI thành danh sách box thống nhất
        boxes_list = sahi_result_to_boxes(sahi_result, class_names)
        image_with_boxes, object_counts = draw_boxes_and_count(image, boxes_list, colors, show_label=show_label)
    else:
        # --- Inference gốc (không SAHI) ---
        results = model.predict(
            source=source_path,
            save=False,
            conf=conf,
            iou=iou_threshold,
            device=model.device,
        )
        image_with_boxes, object_counts = draw_obb_and_count(image, results, class_names, colors, show_label=show_label)
    
    # Thêm lớp hiển thị số đếm lên ảnh (với màu tương ứng)
    final_image = display_counts(image_with_boxes, object_counts, colors=colors)
    
    # Lưu ảnh kết quả trực tiếp vào outputs/images/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(save_dir, f"{Path(source_path).stem}_{timestamp}.jpg")
    cv2.imwrite(output_path, final_image)
    
    return final_image, object_counts, output_path

def predict_video(model, video_path, save_dir_base="outputs", conf=0.6, iou_threshold=0.45,
                  progress_callback=None,
                  use_sahi=True, slice_height=512, slice_width=512,
                  overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                  show_label=False):
    """Dự đoán video với đếm và theo dõi đối tượng.
    
    Khi use_sahi=True (mặc định), mỗi frame được xử lý qua SAHI tiled inference.
    Lưu ý: SAHI không hỗ trợ tracking ID, nên khi bật SAHI sẽ hiển thị số đếm
    per-frame thay vì tracking duy nhất.
    
    Args:
        save_dir_base: Thư mục gốc lưu kết quả (mặc định: 'outputs')
        show_label: Hiển thị nhãn phần trăm trên mỗi box (mặc định: False)
    """
    
    # Tạo thư mục outputs/videos
    save_dir = os.path.join(save_dir_base, "videos")
    os.makedirs(save_dir, exist_ok=True)

    # Mở video đầu vào và lấy thuộc tính
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Thiết lập writer video đầu ra
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(save_dir, f"{Path(video_path).stem}_{timestamp}.mp4")
    fourcc = 0x31637661  # Codec 'avc1' cho định dạng MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Khởi tạo biến theo dõi
    class_names = model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}
    frame_count = 0

    if use_sahi:
        # --- SAHI tiled inference cho video ---
        device = str(model.device)
        model_path = model.ckpt_path
        sahi_model = get_sahi_model(model_path, conf=conf, device=device)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # SAHI hỗ trợ numpy array trực tiếp
            sahi_result = get_sliced_prediction(
                frame,
                sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=iou_threshold,
            )

            boxes_list = sahi_result_to_boxes(sahi_result, class_names)
            frame_with_boxes, frame_counts = draw_boxes_and_count(frame, boxes_list, colors, show_label=show_label)

            final_frame = display_counts(frame_with_boxes, frame_counts, colors=colors)
            out.write(final_frame)
            frame_count += 1

            if progress_callback and total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_callback(progress, frame_count, total_frames)
    else:
        # --- Inference gốc với tracking ---
        tracked_objects = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, conf=conf, iou=iou_threshold, device=model.device, verbose=False)

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
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            class_counts = {cls: list(tracked_objects.values()).count(cls)
                           for cls in set(tracked_objects.values())}
            final_frame = display_counts(frame, class_counts, colors=colors)
            out.write(final_frame)
            frame_count += 1

            if progress_callback and total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_callback(progress, frame_count, total_frames)

    # Dọn dẹp tài nguyên video
    cap.release()
    out.release()
    
    return output_path