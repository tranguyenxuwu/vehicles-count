import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model with CUDA
    results = model.train(
        data="data.yaml",  # Path to your data configuration file
        epochs=100, 
        imgsz=960,
        device=device,
        batch=16,
        workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
        cache=True,
        amp=True,  # Mixed precision for faster training
        optimizer='AdamW',  # Options: 'SGD', 'Adam', 'AdamW', 'RMSProp'
    )

    print(f"Training completed")
    print(f"Results: {results}")