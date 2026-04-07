# Vehicles Count 26

## 📋 Mô tả dự án

Dự án này phát triển các công cụ để huấn luyện, đánh giá và chạy suy luận (inference) mô hình YOLO nhằm phát hiện và đếm phương tiện giao thông. Các mô hình được huấn luyện trên tập dữ liệu **VisDrone** và được đánh giá chuyên sâu đặc biệt với tập dữ liệu **Visdrone_test**. Ứng dụng cung cấp giao diện web thân thiện thông qua Streamlit.

## 🎯 Tính năng chính

- **Phát hiện và đếm đối tượng**: Tự động phát hiện các loại phương tiện với các mô hình YOLO.
- **Hỗ trợ SAHI**: Đánh giá mô hình sử dụng kỹ thuật SAHI bằng cách cắt ảnh (slicing) giúp cải thiện đáng kể khả năng nhận diện phương tiện kích thước nhỏ.
- **Đánh giá mô hình chuyên sâu**: Tích hợp công cụ tính toán các chỉ số AP so sánh trực tiếp với Ground Truth qua các ma trận nhầm lẫn (Confusion Matrix).
- **Giao diện web Streamlit**: Cung cấp dashboard quản lý model, chạy inference trực tiếp trên hình ảnh và video.
- **Chuyển đổi linh hoạt Class**: Hỗ trợ hệ thống tinh lọc tập nhãn linh hoạt đối với nguồn gốc VisDrone (tích hợp loại bỏ các class rác).

## 🏗️ Cấu trúc dự án

```text
vehicles-count-26/
├── streamlit_app.py          # Dashboard Streamlit chạy inference & quản lý model
├── train.py                  # Script huấn luyện các mô hình YOLO
├── inference_utils.py        # Các tiện ích (parse annotations, tính toán IoU, v.v.)
├── split_dataset.py          # Công cụ chia dữ liệu thành tập train/test
├── test/
│   ├── evaluate_models.py    # Script đánh giá mô hình tiêu chuẩn
│   ├── model/                # Chứa các mô hình đã được dùng để test (yolo26m, yolo26m-p2)
│   └── Visdrone_test/        # Tập hình ảnh và Ground truth (YOLO format) phục vụ việc test
├── models/                   # Thư mục lưu mặc định các file model `.pt` cho Streamlit app
└── data.yaml                 # Cấu hình chứa thông tin dataset
```

## 🚀 Cài đặt môi trường

Dự án khuyến nghị chạy trong môi trường **Conda** có tên là `cvenv`, chứa các thư viện cốt lõi như `ultralytics`, `opencv-python`, `numpy`.

### 1. Clone repository

```bash
git clone https://github.com/tranguyenxuwu/vehicles-count.git
cd vehicles-count-26
```

### 2. Khởi tạo môi trường Conda (`cvenv`)

```bash
conda create -n cvenv python=3.9 -y
conda activate cvenv
```

### 3. Cài đặt Cấu hình & Thư viện

Cài đặt PyTorch tương thích với hệ thống (Ví dụ sau cho hệ có GPU NVIDIA / CUDA 12.x):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Cài đặt các gói hiển thị và machine learning khác:

```bash
pip install -r requirements.txt
```

## 🎮 Hướng dẫn sử dụng

Mọi lệnh chạy script nên được sử dụng bên trong môi trường `cvenv` đã được thiết lập.

### 1. Triển khai Streamlit Dashboard

Khởi chạy ứng dụng web để quan sát và quản lý:

```bash
conda run -n cvenv streamlit run streamlit_app.py
```
*(Mở trình duyệt tại địa chỉ http://localhost:8501)*

### 2. Đánh giá Mô hình với Visdrone_test 

Thực thi bộ mã kiểm tra chuyên sâu so sánh qua lại trực tiếp giữa dữ liệu dự đoán từ YOLO và hệ thống nhãn YOLO TXT của VisDrone gốc. Hỗ trợ slicing SAHI với ảnh $512 \times 512$ và overlaps $20\%$:

```bash
conda run -n cvenv python test/evaluate_models.py --conf 0.5 --iou 0.5
```

### 3. Huấn luyện Mô hình

```bash
conda run -n cvenv python train.py
```

### 4. Chia Dataset Train/Test

```bash
conda run -n cvenv python split_dataset.py
```

## 📊 Kết quả & Góc nhìn Đánh giá (Insights)

👉 **Xem Báo cáo Đánh giá chi tiết (Native Inference) tại:** [report_native_20260407_210018.md](test/results/report_native_20260407_210018.md)

Dưới đây là một số ghi nhận từ các đợt kiểm tra quy mô lớn với các mô hình vừa được test:

- **`yolo26m`** (Native resolution 960): Thể hiện khả năng tốt ở việc dự đoán chính xác từng class tương tự nhau (bus, car, truck) khi nó có khả năng bắt được toàn bộ các lớp.
- **`yolo26m-p2`** (Native resolution 768): Hoạt động cực kỳ tốt với khả năng nhận diện các vật thể nhỏ (class-agnostic "vehicle"), nhưng nó có thể gặp hạn chế với class `bus` do mô hình này chỉ được làm quen với 5-class subset trong quá trình huấn luyện thay vì nhóm nhiều class. 
- **Đánh giá Trực tiếp (Native Validating)**: Vì kiểm thử trực tiếp trên cùng định dạng tập con của `Visdrone_test`, các mô hình cho ra quỹ đạo đo lường ổn định nguyên bản (không bị dối loạn Precision do khác biệt định dạng dán nhãn chéo tập). Hàm Loss bám sát các luồng giao thông hỗn hợp.

## 💡 Q&A: Câu hỏi thường gặp Kỹ thuật (Trung bình - Khó)

**1. Khó khăn lớn nhất khi đánh giá model YOLO trên tập `Visdrone_test` nguyên bản là gì?**
VisDrone là một tập dữ liệu nổi tiếng về mật độ giao thông ("dense traffic"). Mật độ dán nhãn (Ground Truth) chằng chịt lên nhau và tỷ lệ che khuất chéo (occlusion) cực cao khiến cho việc điều tiết thuật toán tối ưu là một bài toán hóc búa. Sai số lớn nhất thường rơi vào giới hạn phân biệt mảng pixel mép xe trong đám đông di chuyển liên tục, đòi hỏi hàm NMS không vô tình cắt nhầm tín hiệu của chiếc xe kề cận.

**2. Kỹ thuật SAHI (Slicing Aided Hyper Inference) hoạt động như thế nào trong dự án?**
SAHI giúp YOLO nhận diện các phương tiện có kích thước cực nhỏ trên ảnh phân giải gốc cao chụp từ UAV/Drone. Thay vì nội suy resize toàn bộ ảnh (làm mất pixel của vật nhỏ), dự án cắt ảnh lớn thành các mảnh (slice) $512 \times 512$ có đè lên nhau (overlap $20\%$). Mô hình dự đoán qua từng mảnh, sau cùng áp dụng thuật toán NMS (Non-Maximum Suppression) trên hệ tọa độ gốc để gộp các bounding box lại thành kết quả cuối nhằm tránh trùng lặp.

**3. Tại sao mô hình `yolo26m-p2` lại khả thi hơn cho class-agnostic "vehicle" mặc dù thiếu class "bus"?**
Hậu tố `-p2` đại diện cho việc kiến trúc này được cấu hình thêm một Detection Head P2 đặc biệt kết nối với các Feature Map độ phân giải cao ở nhánh nông của mạng. Việc rễ nhánh giúp nó vượt trội trong việc bắt các vật thể li ti. Kể cả khi có thể mô hình này đã sử dụng tệp huấn luyện bỏ sót lớp `bus` (VD subset 5-class), độ chính xác bắt "vật nhỏ" siêu hạng của nhánh P2 khiến số lượng tổng xe bắt được (class-agnostic) trội hơn các bản tiêu chuẩn.

**4. Quản lý Class Mapping khi trực tiếp đánh giá trên `Visdrone_test` quy mô đầy đủ sẽ diễn ra như thế nào?**
Do đánh giá trực tiếp trên `Visdrone_test`, bộ script không cần phải ép nhãn (map class) về các định dạng thu gọn cục bộ nữa. Dù vậy, đối với những nhãn gốc không thật sự mang tính quyết định đến luồng cơ giới trọng điểm (như `pedestrian` hay `people`), cơ chế ánh xạ có thể mask tắt đo lường chúng trên chuỗi Tensors tính toán mAP nhằm tập trung tuyệt đối vào tập xe tĩnh và xe động (car, bus, truck).

**5. Trình đánh giá YOLO trên `Visdrone_test` ưu ái format Ground Truth TXT như thế nào so với bộ Supervisely?**
Nhờ thư mục `Visdrone_test` vốn đã được biên dịch hệ nhãn về dạng YOLO (`class_id`, `center_x`, `center_y`, `width`, `height`), việc giữ nguyên luồng đọc TXT tinh giản này cho phép Evaluator vòng qua được khâu xử lý cấp CPU nặng nề - vốn dĩ phải xuất hiện nếu phải parse cây đối tượng đa giác JSON của chuẩn ngoại vi. Từ đó đẩy nhanh tốc độ Dataloader qua hàng loạt ma trận ảnh phân giải cao.

**6. Tại sao độ phân giải Inference gốc (Native Resolution) của `yolo26m` (960) lại lớn hơn `yolo26m-p2` (768)?**
`yolo26m` dùng ma trận biên dịch input size lến đến 960 nhằm cố gắng đảm bảo mật độ chi tiết tối ưu trước khi bị downsample mạng backbone. Trong khi đó, `yolo26m-p2` đã xử lý bù đắp kích thước bằng kiến trúc `P2 head` từ thuật toán thiết kế, nên nó chỉ cần input size 768 là đã có đủ vùng tiếp nhận thông tin (Receptive Field) của vật thể bé xíu, qua đó giúp tăng FPS suy luận và tiết kiệm VRAM hơn đáng kể so với việc phải tăng input resolution.

**7. Ngưỡng `--iou 0.5` và `--conf 0.5` trong lệnh chạy `evaluate_models.py` kiểm soát chéo kết quả như thế nào?**
`--conf 0.5` (Confidence Threshold) đóng vai trò giữ cổng, loại bỏ các cụm pixel mà xác suất phán đoán của mô hình là dưới 50% nhằm triệt tiêu báo động giả (False Positive). `--iou 0.5` (IoU / NMS Threshold) quyết định mức độ xóa bỏ các Bounding Box chồng mép lên nhau quanh cùng một chiếc xe. Sự kết hợp này là bắt buộc: `--conf` lọc nhiễu ban đầu, trong khi tối ưu `--iou` đảm bảo thuật toán không vô tình gộp nhầm các chiếc xe đang di chuyển san sát hoặc kẹt xe trên đường thành một vật thể duy nhất.

**8. Tầm quan trọng của việc duy trì sự nhất quán giữa tập Huấn Luyện (VisDrone) và Đánh Giá (`Visdrone_test`)?**
Đặc điểm quay của thiết bị Drone có độ biến thiên rất cao về cực phối cảnh và thu nhỏ góc nhìn chim bay (bird-eye view). Bằng cách đánh giá các hệ trọng số thẳng trên nền `Visdrone_test` thuần túy, chúng ta bảo toàn tính trọn vẹn của Distribution, không làm rỗng rãnh gradient của mô hình bởi các nhiễu loạn domain chéo bóp méo hình dạng (như khi đánh giá chéo UAVDT). Nhờ vậy, kết quả mAP (mean Average Precision) cho ra phản ánh chính xác nhất tính hiệu quả thực tế của mạng hội chập sâu YOLO.

**9. Cách tính toán chỉ số AP (Average Precision) phải tự điều chỉnh ra sao khi gặp tình trạng mâu thuẫn Overpredict?**
Một khi có sự xuất hiện số lượng cực lớn các tín hiệu True Positive nhưng Ground Truth lại bác bỏ nhãn do không dán, nó lập tức nảy lên False Positive cực tay. Điều này kéo sụp vạch ngang đường con Precision, lúc này Recall thì cực cao nhưng mAP tính ra diện tích dưới đường PR là vô cùng thê thảm. Đội phát triển khi nhận dạng đặc vụ này phải đổi chiến lược đọc metrics: Không đánh giá thông qua metric PR đơn thuần mà sẽ sử dụng "Recall Absolute" hoặc xem xét phân tập các test cases (test trên bộ Visdrone test/val nội tĩnh) để khẳng định bản chất.

**10. Kiến trúc YOLOv11/YOLO Custom nào thiết lập được tính chất "Real-time Live" Inference trên dashboard Web phân luồng cao?**
Đây là khả năng tới từ lý thuyết xử lý One-Stage (Single-Stage Detector); kiến trúc loại hẳn hệ thống kén vành khu vực đặc trưng (Region Proposal Network - RPN có trong R-CNN). Cùng với hệ hệ thống truyền Gradient tinh giản như các khối `C3/C2f/C3k2` và mô hình nhận diện Anchor-Free trực giác định vùng qua trung tâm, YOLO có thể chạy qua một Feed-forward Pass liền mạch cho ra List Tensors Toạ Độ xuất chiếu luồng ảnh thô. Do đó nếu có phần cứng ổn định, nó cho phép truyền luồng video trên Pipeline thời gian thực hoàn toàn trơn tru.

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📞 Thông tin Liên hệ

- Tác giả: [@tranguyenxuwu](https://github.com/tranguyenxuwu)
- Project Link: [https://github.com/tranguyenxuwu/vehicles-count](https://github.com/tranguyenxuwu/vehicles-count)
