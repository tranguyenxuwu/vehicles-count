import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os
import argparse
from inference_utils import predict_image, predict_video

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection")
    parser.add_argument("--image", type=str, help="Path to the image file for prediction.")
    parser.add_argument("--video", type=str, help="Path to the video file for prediction.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "./runs/detect/train6/weights/best.pt"
    model = YOLO(model_path)
    model.to(device)

    if args.image and os.path.exists(args.image):
        img, counts, out_path = predict_image(model, args.image)
        print("Object counts:", counts)
        print("Saved result to:", out_path)
    if args.video and os.path.exists(args.video):
        out_path = predict_video(model, args.video)
        print("Saved result to:", out_path)

if __name__ == '__main__':
    main()