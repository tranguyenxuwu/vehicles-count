# Vehicles Count 26

This repository contains tools for training, evaluating, and running inference with YOLO models for vehicle detection and counting, specifically tested against the UAVDT-preview dataset and trained on VisDrone.

## Project Structure
- `streamlit_app.py`: Streamlit dashboard for running inference and managing models.
- `train.py`: Script for training YOLO models.
- `inference_utils.py`: Utilities for parsing annotations, calculating IoU, etc.
- `split_dataset.py`: Utility to split data into train/test sets.
- `test/evaluate_models.py`: Custom evaluation script for computing AP metrics against Supervisely-formatted JSON annotations.
- `models/`: Directory where trained `.pt` files are conventionally stored for the Streamlit app.
- `test/model/`: Specific models tested recently (`yolo26m`, `yolo26m-p2`).

## Python Environment
The recommended environment for executing scripts is a conda environment named `cvenv`, which should contain `ultralytics`, `opencv-python`, `numpy`, and other standard ML stack dependencies.

## Key Sub-Projects & Workflows

### Model Evaluation on UAVDT-Preview (`test/evaluate_models.py`)
A comprehensive evaluation script was developed to run YOLO inference against a set of images and compare them against bounding boxes defined in Supervisely JSON format. Supports SAHI tiled inference for improved small-object detection.

**Command to run:**
```bash
conda run -n cvenv python test/evaluate_models.py --conf 0.5 --iou 0.5
```

Each model runs at its native resolution (yolo26m: 960, yolo26m-p2: 768) with SAHI slicing (512×512, 20% overlap).

**Recent Key Findings:**
- Models were trained on VisDrone (10 classes) but evaluated on UAVDT-preview (4 classes).
- `yolo26m` performs better on exact class matches (bus, car, truck) because it detects all classes.
- `yolo26m-p2` lacks the "bus" class in its specific model configuration (likely trained on a reduced 5-class subset) but performs slightly better on class-agnostic "vehicle" detection likely due to its P2 small-object head.
- Both models heavily overpredict compared to the UAVDT ground truth, scoring low on absolute metrics because VisDrone trains models to detect very small and partially occluded vehicles that UAVDT ground truth labels ignore.

## Important Context for Future Agents
1. **Class Mappings:** If evaluating a new model, ensure the output classes are being properly mapped to our evaluation classes (`test/evaluate_models.py` has a mapping `VISDRONE_TO_EVAL` dictionary for this logic).
2. **Supervisely Format:** Ground truth in the `test/UAVDT-preview/` directory is stored in `.json` files alongside `.jpg` images, using `points.exterior` multi-point associative format for bounding boxes.
3. **Execution Environment:** When executing scripts on behalf of the user, remember to activate or run inside the `cvenv` conda environment.
