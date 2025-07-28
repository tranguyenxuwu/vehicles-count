# Import các thư viện cần thiết
from inference_utils import predict_image, predict_video  # Import hàm dự đoán ảnh và video
import streamlit as st  # Thư viện tạo web app
from ultralytics import YOLO  # Thư viện YOLO cho object detection
import torch  # PyTorch cho deep learning
import tempfile  # Tạo file tạm thời
import os  # Thao tác với hệ điều hành
from pathlib import Path  # Xử lý đường dẫn file
import base64  # Mã hóa base64
import glob  # Tìm kiếm file theo pattern
import shutil  # Sao chép file

# Hàm đọc model mới nhất từ latest.txt
def get_latest_model():
    """Đọc đường dẫn model mới nhất từ file models/latest.txt"""
    latest_file = Path("models/latest.txt")
    if not latest_file.exists():
        return None
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            model_info = f.read().strip()
        
        # Kiểm tra xem có phải là đường dẫn đầy đủ hay chỉ là tên thư mục
        if model_info:
            # Nếu là tên thư mục (như "20250724"), tạo đường dẫn đầy đủ
            if not model_info.endswith('.pt'):
                model_path = f"models/{model_info}/best.pt"
            else:
                # Nếu đã là đường dẫn đầy đủ
                model_path = model_info
            
            # Kiểm tra xem file model có tồn tại không
            if Path(model_path).exists():
                return model_path
            else:
                return None
        else:
            return None
    except Exception as e:
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
model_option = st.radio(
    "Choose model source:",
    ("Upload model", "Use latest model from models/latest.txt")
)

uploaded_model = None
latest_model_path = None

if model_option == "Upload model":
    # Tạo widget upload file cho model YOLO (.pt file)
    uploaded_model = st.file_uploader("Upload YOLOv11 model (.pt)", type=["pt"])
else:
    # Đọc model mới nhất từ latest.txt
    latest_model_path = get_latest_model()
    if latest_model_path:
        st.success(f"✅ Latest model found: {latest_model_path}")
        # Hiển thị thông tin model từ đường dẫn
        model_name = Path(latest_model_path).parent.name
        st.info(f"📅 Model: {model_name}")
    else:
        st.error("❌ No valid model found in models/latest.txt")

# Tạo widget upload file cho ảnh hoặc video
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Tạo slider để điều chỉnh ngưỡng confidence (độ tin cậy)
conf = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Nút chạy dự đoán
if st.button("Run Predict"):
    # Kiểm tra xem có model và file input không
    has_model = (uploaded_model is not None) or (latest_model_path is not None)
    
    if has_model and uploaded_file is not None:
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
            
            # Xử lý model path
            if uploaded_model is not None:
                # Tạo file tạm thời cho model được upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
                    tmp_model.write(uploaded_model.read())
                    model_path = tmp_model.name
            else:
                # Sử dụng model mới nhất
                model_path = latest_model_path

            # Tạo file tạm thời cho input (ảnh/video)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name

            # Bước 2: Load model YOLO
            status_text.text("🤖 Loading YOLO model...")
            progress_bar.progress(30)
            
            # Kiểm tra device (GPU hoặc CPU)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO(model_path)
            model.to(device)

            # Xử lý ảnh
            if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Bước 3: Dự đoán ảnh
                st.session_state.output_video_path = None

                status_text.text("🖼️ Running image prediction...")
                progress_bar.progress(60)
                
                # Gọi hàm predict_image
                img, counts, out_path = predict_image(model, input_path, conf=conf)
                
                # Bước 4: Lưu ảnh vào thư mục outputs
                status_text.text("💾 Saving image result...")
                progress_bar.progress(80)
                
                # Tạo thư mục outputs nếu chưa có
                outputs_dir = Path("outputs")
                outputs_dir.mkdir(exist_ok=True)
                
                # Sao chép file kết quả vào thư mục outputs với tên duy nhất
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_image_name = f"image_result_{timestamp}.jpg"
                saved_image_path = outputs_dir / saved_image_name
                
                # Sao chép file và đảm bảo nó được lưu
                shutil.copy2(out_path, saved_image_path)
                
                # Lưu đường dẫn vào session state
                st.session_state.output_image_path = str(saved_image_path)
                
                # Bước 5: Hiển thị kết quả
                status_text.text("✅ Displaying results...")
                progress_bar.progress(90)
                
                # Hiển thị ảnh kết quả và số lượng đối tượng
                st.image(str(saved_image_path), caption="Object Detection Result")
                
                # Hiển thị thông tin lưu file
                st.success(f"✅ Image saved successfully: {saved_image_path}")
                
                # Hiển thị số lượng đối tượng
                if counts:
                    st.subheader("Detection Summary:")
                    col1, col2 = st.columns(2)
                    with col1:
                        total_objects = sum(counts.values())
                        st.metric("Total Objects", total_objects)
                    with col2:
                        for class_name, count in counts.items():
                            st.metric(f"{class_name}", count)

                # Xóa file output tạm thời (chỉ xóa file tạm trong predict/)
                if os.path.exists(out_path):
                    os.remove(out_path)
                
                # Bước 6: Hoàn thành
                progress_bar.progress(100)
                status_text.text("🎉 Image processing completed!")

            # Xử lý video
            elif input_path.lower().endswith((".mp4", ".avi", ".mov")):
                # Bước 3: Chuẩn bị xử lý video
                status_text.text("🎬 Preparing video prediction...")
                progress_bar.progress(50)
                
                # Lấy thông tin video để theo dõi tiến trình
                import cv2
                cap = cv2.VideoCapture(input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                out_path = predict_video(model, input_path, conf=conf, progress_callback=update_video_progress)
                
                # Bước 5: Lưu video vào thư mục outputs
                status_text.text("💾 Saving video result...")
                progress_bar.progress(85)
                
                # Tạo thư mục outputs nếu chưa có
                outputs_dir = Path("outputs")
                outputs_dir.mkdir(exist_ok=True)
                
                # Sao chép file kết quả vào thư mục outputs với tên duy nhất
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_video_name = f"video_result_{timestamp}.mp4"
                saved_video_path = outputs_dir / saved_video_name
                
                # Sao chép file và đảm bảo nó được lưu
                shutil.copy2(out_path, saved_video_path)
                
                # Bước 6: Tải video để hiển thị
                status_text.text("📺 Loading processed video...")
                progress_bar.progress(90)
                
                # Lưu đường dẫn video vào session state và hiển thị
                st.session_state.output_video_path = str(saved_video_path)
                with open(saved_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                
                # Hiển thị thông tin lưu file
                st.success(f"✅ Video saved successfully: {saved_video_path}")
                
                # Bước 7: Hoàn thành
                progress_bar.progress(100)
                status_text.text("🎉 Video processing completed!")

        except Exception as e:
            # Xử lý lỗi
            status_text.text(f"❌ Error during prediction: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Dọn dẹp các file tạm thời (chỉ xóa nếu là file upload)
            if uploaded_model is not None and model_path and os.path.exists(model_path):
                os.remove(model_path)
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
    else:
        # Cảnh báo nếu chưa có model hoặc file input
        if not has_model:
            st.warning("Please upload a model or ensure models exist in models/ directory.")
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

# Hiển thị danh sách file đã lưu
if os.path.exists("outputs"):
    files = list(Path("outputs").glob("*"))
    if files:
        st.subheader("📁 Saved Results")
        for file_path in sorted(files, reverse=True)[:10]:  # Hiển thị 10 file gần nhất
            file_size = file_path.stat().st_size / (1024*1024)  # MB
            st.text(f"📄 {file_path.name} ({file_size:.1f} MB)")