# Vehicle Counting Project with YOLOv11

## 📋 Mô tả dự án

Dự án này sử dụng YOLOv11 để phát hiện và đếm phương tiện giao thông (xe ô tô và xe máy) từ ảnh và video. Ứng dụng được xây dựng với Streamlit để tạo giao diện web thân thiện người dùng.

## 🎯 Tính năng chính

- **Phát hiện đối tượng**: Sử dụng YOLOv11 để phát hiện xe ô tô và xe máy
- **Đếm tự động**: Tự động đếm số lượng phương tiện được phát hiện
- **Hỗ trợ đa định dạng**: Xử lý cả ảnh (JPG, JPEG, PNG) và video (MP4, AVI, MOV)
- **Giao diện web**: Streamlit app với thanh tiến trình và hiển thị kết quả real-time
- **Tùy chỉnh confidence**: Slider để điều chỉnh ngưỡng confidence
- **GPU support**: Tự động phát hiện và sử dụng GPU nếu có sẵn

## 🏗️ Cấu trúc dự án

```
count-car-project/
├── streamlit_app.py          # Ứng dụng Streamlit chính
├── inference_utils.py        # Các hàm tiện ích cho inference
├── train.py                 # Script huấn luyện model
├── split_dataset.py         # Script chia dataset
├── data.yaml               # Cấu hình dataset
├── requirements.txt        # Các thư viện cần thiết
├── yolo11n.pt             # Model YOLOv11 pretrained
├── models/               # Các model đã huấn luyện
```

## 🚀 Cài đặt và chạy dự án

### 1. Clone repository

```bash
git clone https://github.com/tranguyenxuwu/vehicles-count.git
cd count-car-project
```

### 2. Cài đặt PyTorch

Cài đặt PyTorch phù hợp với hệ thống của bạn:

**Với GPU NVIDIA:**

Chạy lệnh sau để kiểm tra tương thích CUDA

```bash
nvidia-smi
```

Nó nên có output như sau, có thể khác một chút tùy vào GPU của bạn

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.16                 Driver Version: 572.16         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
```

Sau khi chắc chắn tương thích CUDA, chạy lệnh sau để cài đặt PyTorch tương thích

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Hoặc truy cập [pytorch.org](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt phù hợp với hệ thống của bạn.

### 3. Cài đặt các dependencies khác

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng Streamlit

```bash
streamlit run streamlit_app.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

## 📦 Dependencies

- **streamlit**: Tạo giao diện web
- **ultralytics**: YOLOv11 framework
- **opencv-python**: Xử lý ảnh và video
- **torch & torchvision**: PyTorch framework
- **pillow**: Xử lý ảnh
- **numpy**: Tính toán khoa học
- **pathlib**: Xử lý đường dẫn file

## 🎮 Hướng dẫn sử dụng

### Sử dụng Streamlit App

1. **Upload Model**: Tải lên file model YOLOv11 (.pt)
2. **Upload File**: Tải lên ảnh hoặc video cần xử lý
3. **Điều chỉnh Confidence**: Sử dụng slider để đặt ngưỡng confidence (0.0 - 1.0)
4. **Chạy dự đoán**: Nhấn nút "Run Predict"
5. **Xem kết quả**: Ảnh/video với các khung bao và số lượng đối tượng

### Huấn luyện Model

```bash
python train.py
```

Các tham số huấn luyện:

- **epochs**: 100
- **image size**: 960x960
- **batch size**: 16
- **optimizer**: AdamW
- **device**: Tự động phát hiện GPU/CPU - Nên dùng GPU để train nhanh hơn

### Chia Dataset

```bash
python split_dataset.py
```

## 🎯 Classes được phát hiện

- **car**: Xe ô tô
- **motorbike**: Xe máy

## 📊 Kết quả

Ứng dụng cung cấp:

- Ảnh/video với các khung bao quanh đối tượng được phát hiện
- Số lượng từng loại phương tiện
- Confidence score cho mỗi detection
- Thanh tiến trình real-time trong quá trình xử lý

## 🔧 Cấu hình

### GPU Support

Dự án tự động phát hiện và sử dụng GPU nếu có:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Dataset Configuration (data.yaml)

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 2 # number of classes
names: ["car", "motorbike"]
```

## 📁 Cấu trúc Data

```
data/
├── raw/                    # Dữ liệu gốc từ Roboflow
├── train_data/            # Dataset đã chia train/val
│   └── 250720/
│       ├── train/
│       └── val/
├── test/                  # Test dataset
│   ├── images/
│   └── labels/
└── test_videos/          # Video test
```

## 🚀 Các tính năng nâng cao

- **Progress tracking**: Thanh tiến trình chi tiết cho video processing
- **Error handling**: Xử lý lỗi và cleanup tự động
- **Temporary files**: Quản lý file tạm thời an toàn
- **Session state**: Lưu trạng thái giữa các lần chạy

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📞 Liên hệ

- GitHub: [@tranguyenxuwu](https://github.com/tranguyenxuwu)
- Project Link: [https://github.com/tranguyenxuwu/vehicles-count](https://github.com/tranguyenxuwu/vehicles-count)

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

⭐ Nếu dự án hữu ích, hãy cho một star nhé!
