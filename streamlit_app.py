# Import các thư viện cần thiết
from inference_utils import predict_image, predict_video  # Import hàm dự đoán ảnh và video
import streamlit as st  # Thư viện tạo web app
from ultralytics import YOLO  # Thư viện YOLO cho object detection
import torch  # PyTorch cho deep learning
import tempfile  # Tạo file tạm thời
import os  # Thao tác với hệ điều hành
from pathlib import Path  # Xử lý đường dẫn file
import time  # Đo thời gian xử lý

# Hàm lấy danh sách model có sẵn
def get_available_models():
    """Quét thư mục models/ và trả về danh sách tên model (thư mục chứa best.pt)"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    return sorted(
        [d.name for d in models_dir.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    )

# Hàm đọc model mặc định từ latest.txt
def get_default_model():
    """Đọc tên model mặc định từ file models/latest.txt"""
    latest_file = Path("models/latest.txt")
    if not latest_file.exists():
        return None
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip() or None
    except Exception:
        return None

# Tạo tiêu đề cho ứng dụng web
st.title("YOLOv11 Traffic Detection App")

# Khởi tạo session state để lưu đường dẫn output
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'output_image_path' not in st.session_state:
    st.session_state.output_image_path = None

# Tùy chọn model
st.subheader("Model Selection")
available_models = get_available_models()
default_model_name = get_default_model()

latest_model_path = None

if not available_models:
    st.error("❌ No models found in models/ directory (each model folder must contain best.pt)")
else:
    # Xác định index mặc định
    default_index = 0
    if default_model_name and default_model_name in available_models:
        default_index = available_models.index(default_model_name)

    selected_model = st.selectbox("Select a model:", available_models, index=default_index)
    latest_model_path = f"models/{selected_model}/best.pt"
    st.success(f"✅ Using model: {latest_model_path}")

# Tạo widget upload file cho ảnh hoặc video
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Tạo slider để điều chỉnh ngưỡng confidence (độ tin cậy)
conf = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
iou = st.slider("IoU threshold (NMS)", min_value=0.0, max_value=1.0, value=0.45, step=0.05, 
                help="Intersection over Union threshold. Lower values remove more overlapping boxes for the same object.")

# --- SAHI Tiled Inference Settings ---
with st.sidebar:
    st.header("⚙️ SAHI Tiled Inference")
    use_sahi = st.checkbox("Enable SAHI", value=True,
                           help="Slices the image into tiles for better small-object detection.")
    if use_sahi:
        slice_height = st.slider("Slice Height (px)", min_value=128, max_value=1024, value=512, step=64)
        slice_width = st.slider("Slice Width (px)", min_value=128, max_value=1024, value=512, step=64)
        overlap_ratio = st.slider("Overlap Ratio", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                                  help="Fraction of overlap between adjacent tiles.")
    else:
        slice_height = 512
        slice_width = 512
        overlap_ratio = 0.2

    st.divider()
    st.header("🏷️ Display Options")
    show_label = st.checkbox("Show labels on detections", value=False,
                             help="Show class name and confidence percentage on each bounding box. "
                                  "Disabled by default to avoid covering small objects.")

# Nút chạy dự đoán
if st.button("Run Predict"):
    # Kiểm tra xem có model và file input không
    has_model = latest_model_path is not None
    
    if has_model and uploaded_file is not None:
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Tạo thanh tiến trình và text trạng thái
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Khởi tạo biến đường dẫn
        model_path = None
        input_path = None
        
        try:
            # Bước 1: Lưu các file đã upload
            status_text.text("📁 Preparing files...")
            progress_bar.progress(10)
            
            # Sử dụng model đã chọn
            model_path = latest_model_path

            # Tạo file tạm thời cho input (ảnh/video)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name

            # Bước 2: Load model YOLO
            status_text.text("🤖 Loading YOLO model...")
            progress_bar.progress(30)
            
            # Kiểm tra device (GPU, MPS hoặc CPU)
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            model = YOLO(model_path)
            model.to(device)

            # Xử lý ảnh
            if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Bước 3: Dự đoán ảnh
                st.session_state.output_video_path = None

                status_text.text("🖼️ Running image prediction...")
                progress_bar.progress(60)
                
                # Gọi hàm predict_image
                img, counts, out_path = predict_image(
                    model, input_path, conf=conf, iou_threshold=iou,
                    use_sahi=use_sahi,
                    slice_height=slice_height, slice_width=slice_width,
                    overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
                    show_label=show_label,
                )
                
                # Bước 4: Kết quả đã được lưu trực tiếp vào outputs/images/
                status_text.text("💾 Result saved...")
                progress_bar.progress(80)
                
                # Lưu đường dẫn vào session state
                st.session_state.output_image_path = out_path
                
                # Bước 5: Hiển thị kết quả
                status_text.text("✅ Displaying results...")
                progress_bar.progress(90)
                
                # Tính tổng thời gian xử lý
                end_time = time.time()
                total_time = end_time - start_time
                
                # Hiển thị ảnh kết quả và số lượng đối tượng
                st.image(out_path, caption="Object Detection Result")
                
                # Hiển thị thông tin lưu file
                st.success(f"✅ Image saved: {out_path}")
                
                # Hiển thị số lượng đối tượng và thời gian xử lý
                if counts:
                    st.subheader("Detection Summary:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_objects = sum(counts.values())
                        st.metric("Total Objects", total_objects)
                    with col2:
                        st.metric("Processing Time", f"{total_time:.2f}s")
                    with col3:
                        if device == 'cuda':
                            device_info = "GPU (CUDA)"
                        elif device == 'mps':
                            device_info = "Apple MPS"
                        else:
                            device_info = "CPU"
                        st.metric("Device Used", device_info)
                    
                    # Hiển thị chi tiết từng loại đối tượng
                    st.subheader("Object Details:")
                    detail_cols = st.columns(len(counts))
                    for idx, (class_name, count) in enumerate(counts.items()):
                        with detail_cols[idx]:
                            st.metric(f"{class_name}", count)
                
                # Bước 6: Hoàn thành
                progress_bar.progress(100)
                status_text.text(f"🎉 Image processing completed in {total_time:.2f}s!")

            # Xử lý video
            elif input_path.lower().endswith((".mp4", ".avi", ".mov")):
                # Bước 3: Chuẩn bị xử lý video
                status_text.text("🎬 Preparing video prediction...")
                progress_bar.progress(50)
                
                # Lấy thông tin video để theo dõi tiến trình
                import cv2
                cap = cv2.VideoCapture(input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                # Tạo thanh tiến trình con cho xử lý video
                video_progress_container = st.container()
                with video_progress_container:
                    st.write("Video processing progress:")
                    video_progress = st.progress(0)
                    frame_info = st.empty()
                
                # Hàm callback cập nhật tiến trình video
                def update_video_progress(progress, current_frame, total):
                    video_progress.progress(progress)
                    frame_info.text(f"Processing frame {current_frame}/{total} ({progress*100:.1f}%)")
                    # Cập nhật thanh tiến trình chính (từ 60% đến 80%)
                    main_progress = 60 + (progress * 20)
                    progress_bar.progress(int(main_progress))
                
                # Bước 4: Dự đoán video với cập nhật tiến trình
                status_text.text(f"🎥 Processing video ({total_frames} frames)...")

                # Gọi hàm predict_video với callback
                out_path = predict_video(
                    model, input_path, conf=conf, iou_threshold=iou,
                    progress_callback=update_video_progress,
                    use_sahi=use_sahi,
                    slice_height=slice_height, slice_width=slice_width,
                    overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
                    show_label=show_label,
                )
                
                # Bước 5: Kết quả đã được lưu trực tiếp vào outputs/videos/
                status_text.text("💾 Result saved...")
                progress_bar.progress(85)
                
                # Bước 6: Tải video để hiển thị
                status_text.text("📺 Loading processed video...")
                progress_bar.progress(90)
                
                # Tính tổng thời gian xử lý
                end_time = time.time()
                total_time = end_time - start_time
                
                # Lưu đường dẫn video vào session state và hiển thị
                st.session_state.output_video_path = out_path
                with open(out_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                
                # Hiển thị thông tin lưu file
                st.success(f"✅ Video saved: {out_path}")
                
                # Hiển thị thông tin video
                st.subheader("Video Processing Summary:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", total_frames)
                with col2:
                    st.metric("Processing Time", f"{total_time:.2f}s")
                with col3:
                    st.metric("Video Duration", f"{duration:.1f}s")
                with col4:
                    processing_speed = total_frames / total_time if total_time > 0 else 0
                    st.metric("Speed", f"{processing_speed:.1f} FPS")
                
                # Hiển thị thông tin thiết bị
                if device == 'cuda':
                    device_info = "GPU (CUDA)"
                elif device == 'mps':
                    device_info = "Apple MPS"
                else:
                    device_info = "CPU"
                st.info(f"🖥️ Processed on: {device_info}")
                
                # Bước 7: Hoàn thành
                progress_bar.progress(100)
                status_text.text(f"🎉 Video processing completed in {total_time:.2f}s!")

        except Exception as e:
            # Xử lý lỗi
            status_text.text(f"❌ Error during prediction: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Dọn dẹp các file tạm thời (chỉ xóa nếu là file upload)

            if input_path and os.path.exists(input_path):
                os.remove(input_path)
    else:
        # Cảnh báo nếu chưa có model hoặc file input
        if not has_model:
            st.warning("Please select a model from the list above.")
        if uploaded_file is None:
            st.warning("Please upload an image or video file.")

# Phần download kết quả (hiển thị sau khi đã có kết quả)
st.divider()
st.subheader("Download Results")

# Download ảnh kết quả
if st.session_state.output_image_path and os.path.exists(st.session_state.output_image_path):
    with open(st.session_state.output_image_path, "rb") as file:
        st.download_button(
            label="📥 Download Image Result",
            data=file.read(),
            file_name=Path(st.session_state.output_image_path).name,
            mime="image/jpeg"
        )

# Download video kết quả
if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
    with open(st.session_state.output_video_path, "rb") as file:
        st.download_button(
            label="📥 Download Video Result",
            data=file.read(),
            file_name=Path(st.session_state.output_video_path).name,
            mime="video/mp4"
        )

# Hiển thị danh sách file đã lưu (images + videos)
outputs_root = Path("outputs")
if outputs_root.exists():
    all_files = sorted(
        list((outputs_root / "images").glob("*")) + list((outputs_root / "videos").glob("*")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if all_files:
        st.subheader("📁 Saved Results")
        for file_path in all_files[:10]:  # Hiển thị 10 file gần nhất
            file_size = file_path.stat().st_size / (1024*1024)  # MB
            category = file_path.parent.name  # 'images' or 'videos'
            st.text(f"📄 [{category}] {file_path.name} ({file_size:.1f} MB)")