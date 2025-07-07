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
        imgsz=640,
        device=device,
        batch=32,
        workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
        cache=True,
        amp=True,  # Mixed precision for faster training
        optimizer='AdamW',  # Options: 'SGD', 'Adam', 'AdamW', 'RMSProp'
        lr0=0.005,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1  # Warmup initial bias learning rate
    )

    print(f"Training completed")
    print(f"Results: {results}")