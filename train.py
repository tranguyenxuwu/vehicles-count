from ultralytics import YOLO

# Khởi tạo mô hình
model = YOLO("yolo26m")

results = model.train(
    data="./visdrone/dataset.yaml",
    device="cuda",
    resume=False,
    epochs=100,
    patience=20,
    imgsz=960,
    lr0=0.001,
    batch=6,
    optimizer="MuSGD",
    amp=True,
    cache=True,
    workers=16,         
    close_mosaic=15, 
    mixup=0.1,
    copy_paste=0.1    
)
