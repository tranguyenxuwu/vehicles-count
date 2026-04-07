#!/usr/bin/env python3
"""
YOLO Model Evaluation on UAVDT-Preview Dataset (SAHI)
=====================================================
Evaluates all YOLO models in test/model/ against UAVDT-preview images
(train + test splits combined), using existing Supervisely JSON labels
as ground truth.

Uses SAHI (Slicing Aided Hyper Inference) for improved small-object
detection.  Each model runs at its native resolution:
  • yolo26m    → 960 px
  • yolo26m-p2 → 768 px

Metrics computed:
  - Per-class AP@50, AP@50:95, Precision, Recall
  - Overall mAP@50, mAP@50:95
  - Inference speed (FPS)
  - Vehicle counting accuracy (MAE, RMSE per image)
  - Confusion-style analysis

Usage:
    conda activate cvenv
    python evaluate_models.py [--conf 0.5] [--iou 0.5]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─── PATHS ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "Visdrone_test"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── PER-MODEL NATIVE RESOLUTION ──────────────────────────────────────
MODEL_NATIVE_IMGSZ = {
    "yolo26m":    960,
    "yolo26m-p2": 768,
}

# ─── SAHI DEFAULTS ─────────────────────────────────────────────────────
SAHI_SLICE_H = 512
SAHI_SLICE_W = 512
SAHI_OVERLAP_H = 0.2
SAHI_OVERLAP_W = 0.2

# ─── CLASS MAPPING ─────────────────────────────────────────────────────
# UAVDT ground-truth classes from meta.json
UAVDT_CLASSES = ["bus", "car", "truck", "vehicle"]

# Name-based mapping from model class names → unified evaluation class name.
# Works for both old 10-class VisDrone and new 5-class reduced dataset.
# We merge car+van → car, and treat "vehicle" labels as a generic match.
NAME_TO_EVAL = {
    "car":   "car",
    "van":   "car",   # van → car
    "truck": "truck",
    "bus":   "bus",
}

# For the "vehicle" GT class (found in test split), any of these pred classes match
VEHICLE_PRED_CLASSES = {"car", "truck", "bus"}


def build_cls_id_to_eval(model_names: dict) -> dict:
    """Build a mapping from model class index → eval class name at runtime."""
    mapping = {}
    for cls_id, cls_name in model_names.items():
        name_lower = cls_name.lower()
        if name_lower in NAME_TO_EVAL:
            mapping[cls_id] = NAME_TO_EVAL[name_lower]
    return mapping


def parse_yolo_txt(txt_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    Parse a YOLO .txt annotation file.
    Returns list of dicts: {class_name, bbox: [x1,y1,x2,y2]}
    """
    # The Visdrone_test YOLO labels are produced for the 5-class reduced dataset
    VISDRONE_CLASSES_LIST = [
        "pedestrian", "car", "truck", "bus", "motor"
    ]
    
    annotations = []
    if not txt_path.exists():
        return annotations
        
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id >= len(VISDRONE_CLASSES_LIST):
                    continue
                cls_name = VISDRONE_CLASSES_LIST[cls_id]
                
                eval_cls = cls_name
                
                if eval_cls not in ["car", "truck", "bus"]:
                    continue
                
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                box_w = float(parts[3]) * img_w
                box_h = float(parts[4]) * img_h
                
                x1 = x_center - box_w / 2
                y1 = y_center - box_h / 2
                x2 = x_center + box_w / 2
                y2 = y_center + box_h / 2
                
                annotations.append({
                    "class_name": eval_cls,
                    "bbox": [x1, y1, x2, y2],
                })
    return annotations


def collect_dataset() -> list[dict]:
    """
    Collect all images with their ground truth.
    Returns list of {img_path, ann_path, split, gt_boxes: [...]}
    """
    dataset = []
    for split in ["test"]:
        img_dir = DATASET_DIR / "images" / split
        ann_dir = DATASET_DIR / "labels" / split
        if not img_dir.exists():
            print(f"  ⚠ Split '{split}' not found, skipping...")
            continue

        img_files = sorted(img_dir.glob("*.jpg"))
        for img_path in img_files:
            ann_path = ann_dir / f"{img_path.stem}.txt"
            
            gt_boxes = []
            if ann_path.exists():
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gt_boxes = parse_yolo_txt(ann_path, w, h)

            if len(gt_boxes) == 0:
                continue

            dataset.append({
                "img_path": str(img_path),
                "ann_path": str(ann_path),
                "split": split,
                "gt_boxes": gt_boxes,
            })

    return dataset


# ─── IoU & AP COMPUTATION ─────────────────────────────────────────────

def compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precision_list, recall_list):
    """Compute Average Precision using 101-point interpolation (COCO-style)."""
    if len(recall_list) == 0:
        return 0.0

    # Add sentinel values
    mrec = [0.0] + list(recall_list) + [1.0]
    mpre = [0.0] + list(precision_list) + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_rec = 0.0
        for i in range(len(mrec)):
            if mrec[i] >= t:
                prec_at_rec = mpre[i]
                break
        ap += prec_at_rec / 101.0

    return ap


def evaluate_detections(all_gt, all_det, iou_threshold=0.5):
    """
    Evaluate detections against ground truth for a single class.
    
    Args:
        all_gt: list of (img_idx, bbox)
        all_det: list of (img_idx, confidence, bbox)
        iou_threshold: IoU threshold for matching
    
    Returns: dict with AP, precision, recall
    """
    n_gt = len(all_gt)
    if n_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": 0, "n_det": len(all_det)}

    # Sort detections by confidence (descending)
    all_det_sorted = sorted(all_det, key=lambda x: x[1], reverse=True)

    # Group GT by image index
    gt_by_img = defaultdict(list)
    for img_idx, bbox in all_gt:
        gt_by_img[img_idx].append({"bbox": bbox, "matched": False})

    tp_list = []
    fp_list = []

    for det_img_idx, det_conf, det_bbox in all_det_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_item in enumerate(gt_by_img.get(det_img_idx, [])):
            iou = compute_iou(det_bbox, gt_item["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_item = gt_by_img[det_img_idx][best_gt_idx]
            if not gt_item["matched"]:
                gt_item["matched"] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                # Duplicate detection
                tp_list.append(0)
                fp_list.append(1)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Compute cumulative precision/recall
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    recall_curve = tp_cumsum / n_gt
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap(precision_curve, recall_curve)

    # Final precision/recall at all detections
    total_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
    total_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
    final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    final_recall = total_tp / n_gt if n_gt > 0 else 0.0

    return {
        "AP": round(ap, 4),
        "precision": round(final_precision, 4),
        "recall": round(final_recall, 4),
        "n_gt": n_gt,
        "n_det": len(all_det),
        "TP": total_tp,
        "FP": total_fp,
        "FN": n_gt - total_tp,
    }


# ─── COUNTING METRICS ─────────────────────────────────────────────────

def compute_counting_metrics(per_image_gt_counts, per_image_det_counts):
    """Compute MAE, RMSE, and R² for vehicle counting."""
    gt = np.array(per_image_gt_counts, dtype=float)
    det = np.array(per_image_det_counts, dtype=float)
    errors = det - gt

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "mean_gt_count": round(float(np.mean(gt)), 2),
        "mean_det_count": round(float(np.mean(det)), 2),
        "median_error": round(float(np.median(errors)), 2),
    }


# ─── MAIN EVALUATION ──────────────────────────────────────────────────

def map_prediction_class(yolo_cls_id: int, cls_id_to_eval: dict) -> str | None:
    """Map a YOLO prediction class ID to our eval class name."""
    return cls_id_to_eval.get(yolo_cls_id)


def _map_sahi_class(pred_obj, cls_id_to_eval: dict) -> str | None:
    """Map a SAHI prediction object to our eval class name."""
    cls_id = pred_obj.category.id
    return cls_id_to_eval.get(cls_id)


def run_evaluation(model_name: str, model_path: str, dataset: list[dict],
                   conf: float, iou_thresh: float, imgsz: int) -> dict:
    """Run full evaluation for one model using SAHI tiled inference."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}  (SAHI)")
    print(f"  Model:      {model_path}")
    print(f"  Images:     {len(dataset)}")
    print(f"  Conf:       {conf}  |  IoU:  {iou_thresh}  |  ImgSz: {imgsz}")
    print(f"  SAHI slice: {SAHI_SLICE_H}×{SAHI_SLICE_W}  "
          f"overlap: {SAHI_OVERLAP_H}/{SAHI_OVERLAP_W}")
    print(f"{'='*70}")

    # Load YOLO model (for class names reference)
    yolo_model = YOLO(model_path)
    model_names = yolo_model.names
    cls_id_to_eval = build_cls_id_to_eval(model_names)
    print(f"  Model classes: {model_names}")
    print(f"  Eval mapping:  {cls_id_to_eval}")

    # Build SAHI detection model
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="mps",  # Apple Silicon GPU
        image_size=imgsz,
    )

    # Storage for per-class evaluations
    eval_classes = ["bus", "car", "truck"]
    gt_by_class = {c: [] for c in eval_classes}
    det_by_class = {c: [] for c in eval_classes}

    gt_vehicle = []
    det_vehicle = []

    per_image_gt_counts = []
    per_image_det_counts = []

    split_stats = {"train": {"n": 0, "gt": 0, "det": 0},
                   "test": {"n": 0, "gt": 0, "det": 0}}

    total_inference_time = 0.0
    n_processed = 0

    for img_idx, sample in enumerate(dataset):
        img_path = sample["img_path"]
        gt_boxes = sample["gt_boxes"]
        split = sample["split"]

        # ── Run SAHI sliced inference ──
        t0 = time.time()
        sahi_result = get_sliced_prediction(
            img_path,
            sahi_model,
            slice_height=imgsz,
            slice_width=imgsz,
            overlap_height_ratio=SAHI_OVERLAP_H,
            overlap_width_ratio=SAHI_OVERLAP_W,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=iou_thresh,
            verbose=0,
        )
        t1 = time.time()
        total_inference_time += (t1 - t0)
        n_processed += 1

        # ── Parse SAHI predictions ──
        preds = []
        for pred_obj in sahi_result.object_prediction_list:
            eval_cls = _map_sahi_class(pred_obj, cls_id_to_eval)
            if eval_cls:
                bbox = pred_obj.bbox
                preds.append({
                    "class_name": eval_cls,
                    "confidence": pred_obj.score.value,
                    "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                })

        # ── Assign GT and predictions to class buckets ──
        n_gt_this = len(gt_boxes)
        n_det_this = len(preds)

        for gt in gt_boxes:
            gt_cls = gt["class_name"]
            if gt_cls == "vehicle":
                gt_vehicle.append((img_idx, gt["bbox"]))
            elif gt_cls in eval_classes:
                gt_by_class[gt_cls].append((img_idx, gt["bbox"]))
                gt_vehicle.append((img_idx, gt["bbox"]))

        for pred in preds:
            pred_cls = pred["class_name"]
            if pred_cls in eval_classes:
                det_by_class[pred_cls].append(
                    (img_idx, pred["confidence"], pred["bbox"]))
            det_vehicle.append((img_idx, pred["confidence"], pred["bbox"]))

        per_image_gt_counts.append(n_gt_this)
        per_image_det_counts.append(n_det_this)
        split_stats[split]["n"] += 1
        split_stats[split]["gt"] += n_gt_this
        split_stats[split]["det"] += n_det_this

        # ── Progress ──
        if (img_idx + 1) % 100 == 0 or img_idx == len(dataset) - 1:
            fps = n_processed / total_inference_time if total_inference_time > 0 else 0
            print(f"  [{img_idx+1:4d}/{len(dataset)}]  "
                  f"FPS: {fps:.1f}  |  GT: {n_gt_this:3d}  Det: {n_det_this:3d}")

    # ── Compute per-class metrics ──
    print(f"\n  Computing metrics...")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    per_class_results = {}
    for cls_name in eval_classes:
        # AP@50
        result_50 = evaluate_detections(gt_by_class[cls_name],
                                        det_by_class[cls_name],
                                        iou_threshold=0.5)
        # AP@50:95
        aps = []
        for iou_t in iou_thresholds:
            r = evaluate_detections(gt_by_class[cls_name],
                                    det_by_class[cls_name],
                                    iou_threshold=iou_t)
            aps.append(r["AP"])
        ap_50_95 = round(float(np.mean(aps)), 4)

        per_class_results[cls_name] = {
            "n_gt": result_50["n_gt"],
            "n_det": result_50["n_det"],
            "AP50": result_50["AP"],
            "AP50_95": ap_50_95,
            "precision_50": result_50["precision"],
            "recall_50": result_50["recall"],
            "TP": result_50["TP"],
            "FP": result_50["FP"],
            "FN": result_50["FN"],
        }

    # ── "All Vehicles" evaluation ──
    result_vehicle_50 = evaluate_detections(gt_vehicle, det_vehicle, iou_threshold=0.5)
    aps_vehicle = []
    for iou_t in iou_thresholds:
        r = evaluate_detections(gt_vehicle, det_vehicle, iou_threshold=iou_t)
        aps_vehicle.append(r["AP"])
    ap_vehicle_50_95 = round(float(np.mean(aps_vehicle)), 4)

    # ── Overall mAP (average across non-empty classes) ──
    valid_aps_50 = [v["AP50"] for v in per_class_results.values() if v["n_gt"] > 0]
    valid_aps_50_95 = [v["AP50_95"] for v in per_class_results.values() if v["n_gt"] > 0]
    mAP50 = round(float(np.mean(valid_aps_50)), 4) if valid_aps_50 else 0.0
    mAP50_95 = round(float(np.mean(valid_aps_50_95)), 4) if valid_aps_50_95 else 0.0

    # ── Counting metrics ──
    counting = compute_counting_metrics(per_image_gt_counts, per_image_det_counts)

    # ── Compile results ──
    fps = n_processed / total_inference_time if total_inference_time > 0 else 0
    total_gt = sum(per_image_gt_counts)
    total_det = sum(per_image_det_counts)

    result = {
        "model": model_name,
        "model_path": model_path,
        "n_images": len(dataset),
        "conf_threshold": conf,
        "iou_threshold": iou_thresh,
        "imgsz": imgsz,
        "inference_time_s": round(total_inference_time, 2),
        "fps": round(fps, 2),
        "total_gt": total_gt,
        "total_detections": total_det,
        "per_class": per_class_results,
        "all_vehicles": {
            "n_gt": result_vehicle_50["n_gt"],
            "n_det": result_vehicle_50["n_det"],
            "AP50": result_vehicle_50["AP"],
            "AP50_95": ap_vehicle_50_95,
            "precision_50": result_vehicle_50["precision"],
            "recall_50": result_vehicle_50["recall"],
            "TP": result_vehicle_50["TP"],
            "FP": result_vehicle_50["FP"],
            "FN": result_vehicle_50["FN"],
        },
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "counting": counting,
        "split_stats": split_stats,
    }

    # ── Print summary ──
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  Results: {model_name:<36s}│")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  mAP@50:      {mAP50:<10.4f}                   │")
    print(f"  │  mAP@50:95:   {mAP50_95:<10.4f}                   │")
    print(f"  │  FPS:         {fps:<10.1f}                   │")
    print(f"  │  Total GT:    {total_gt:<10d}                   │")
    print(f"  │  Total Det:   {total_det:<10d}                   │")
    print(f"  ├──────────────────────────────────────────────┤")
    for cls_name, cls_res in per_class_results.items():
        if cls_res["n_gt"] > 0:
            print(f"  │  {cls_name:>6s}:  AP50={cls_res['AP50']:.4f}  "
                  f"P={cls_res['precision_50']:.4f}  "
                  f"R={cls_res['recall_50']:.4f}  │")
    print(f"  │  {'ALL':>6s}:  AP50={result_vehicle_50['AP']:.4f}  "
          f"P={result_vehicle_50['precision']:.4f}  "
          f"R={result_vehicle_50['recall']:.4f}  │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  Counting MAE:  {counting['MAE']:<8.2f}                 │")
    print(f"  │  Counting RMSE: {counting['RMSE']:<8.2f}                 │")
    print(f"  │  Counting R²:   {counting['R2']:<8.4f}                 │")
    print(f"  │  SAHI:          Yes                          │")
    print(f"  │  Slice:         {SAHI_SLICE_H}×{SAHI_SLICE_W}                      │")
    print(f"  └──────────────────────────────────────────────┘")

    return result


def generate_comparison_table(all_results: list[dict]) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append("# YOLO Model Comparison on UAVDT-Preview (SAHI)")
    lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: UAVDT-Preview (train + test splits combined)")
    lines.append(f"Inference: SAHI sliced ({SAHI_SLICE_H}×{SAHI_SLICE_W}, "
                 f"overlap {SAHI_OVERLAP_H}/{SAHI_OVERLAP_W})")
    lines.append("")

    # ── Summary table ──
    lines.append("## Overall Performance")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r["model"] for r in all_results) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")

    metrics = [
        ("Images", "n_images"),
        ("Conf Threshold", "conf_threshold"),
        ("Image Size", "imgsz"),
        ("mAP@50", "mAP50"),
        ("mAP@50:95", "mAP50_95"),
        ("FPS", "fps"),
        ("Inference Time (s)", "inference_time_s"),
        ("Total GT Boxes", "total_gt"),
        ("Total Detections", "total_detections"),
    ]
    for label, key in metrics:
        vals = []
        for r in all_results:
            v = r.get(key, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if v < 1 else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # ── All-vehicles metrics ──
    lines.append("")
    lines.append("## All Vehicles (class-agnostic)")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r["model"] for r in all_results) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")
    for metric_name in ["AP50", "AP50_95", "precision_50", "recall_50", "TP", "FP", "FN"]:
        vals = []
        for r in all_results:
            v = r["all_vehicles"].get(metric_name, 0)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {metric_name} | " + " | ".join(vals) + " |")

    # ── Per-class table ──
    lines.append("")
    lines.append("## Per-Class Performance")
    lines.append("")
    for cls_name in ["bus", "car", "truck"]:
        lines.append(f"### {cls_name.capitalize()}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(r["model"] for r in all_results) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")
        for metric in ["n_gt", "n_det", "AP50", "AP50_95", "precision_50", "recall_50", "TP", "FP", "FN"]:
            vals = []
            for r in all_results:
                v = r["per_class"].get(cls_name, {}).get(metric, 0)
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append(f"| {metric} | " + " | ".join(vals) + " |")
        lines.append("")

    # ── Counting metrics ──
    lines.append("## Vehicle Counting Accuracy")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r["model"] for r in all_results) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")
    for metric in ["MAE", "RMSE", "R2", "mean_gt_count", "mean_det_count", "median_error"]:
        vals = []
        for r in all_results:
            v = r["counting"].get(metric, 0)
            vals.append(f"{v:.4f}" if isinstance(v, float) and v < 1 else f"{v:.2f}")
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    # ── Per-split stats ──
    lines.append("")
    lines.append("## Per-Split Statistics")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| Split | Images | GT Boxes | Detections |")
        lines.append("|-------|--------|----------|------------|")
        for split, stats in r["split_stats"].items():
            lines.append(f"| {split} | {stats['n']} | {stats['gt']} | {stats['det']} |")
        lines.append("")

    return "\n".join(lines)


# ─── MAIN ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO models on UAVDT-Preview dataset (SAHI)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   YOLO Model Evaluation on Visdrone_test Dataset  (SAHI)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Discover models ──
    models = []
    if MODEL_DIR.exists():
        for model_dir in sorted(MODEL_DIR.iterdir()):
            if model_dir.is_dir():
                best_pt = model_dir / "best.pt"
                if best_pt.exists():
                    native_imgsz = MODEL_NATIVE_IMGSZ.get(model_dir.name, 960)
                    models.append({
                        "name": model_dir.name,
                        "path": str(best_pt),
                        "imgsz": native_imgsz,
                    })

    if not models:
        print("❌ No models found in", MODEL_DIR)
        sys.exit(1)

    print(f"\n  Found {len(models)} model(s):")
    for m in models:
        print(f"    • {m['name']}: {m['path']}  (imgsz={m['imgsz']})")

    # ── Collect dataset ──
    print(f"\n  Collecting dataset from {DATASET_DIR}...")
    dataset = collect_dataset()
    n_train = sum(1 for d in dataset if d["split"] == "train")
    n_test = sum(1 for d in dataset if d["split"] == "test")
    total_gt = sum(len(d["gt_boxes"]) for d in dataset)
    print(f"  Total: {len(dataset)} images ({n_train} train + {n_test} test)")
    print(f"  Total GT boxes: {total_gt}")

    # ── Class distribution ──
    class_counts = defaultdict(int)
    for d in dataset:
        for gt in d["gt_boxes"]:
            class_counts[gt["class_name"]] += 1
    print(f"  GT class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:>10s}: {count:5d}")

    # ── Run evaluations ──
    all_results = []
    for m in models:
        result = run_evaluation(
            model_name=m["name"],
            model_path=m["path"],
            dataset=dataset,
            conf=args.conf,
            iou_thresh=args.iou,
            imgsz=m["imgsz"],    # per-model native resolution
        )
        all_results.append(result)

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Individual model results
    for result in all_results:
        out_path = RESULTS_DIR / f"{result['model']}_sahi_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

    # Comparison JSON
    comparison_path = RESULTS_DIR / f"comparison_sahi_{timestamp}.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {comparison_path}")

    # Markdown report
    report = generate_comparison_table(all_results)
    report_path = RESULTS_DIR / f"report_sahi_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # ── Final comparison ──
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON  (SAHI)")
    print(f"{'='*70}")
    print(f"\n  {'Model':<16s} {'ImgSz':>5s} {'mAP@50':>8s} {'mAP@50:95':>10s} "
          f"{'FPS':>6s} {'P':>6s} {'R':>6s} {'MAE':>6s}")
    print(f"  {'-'*16} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in all_results:
        print(f"  {r['model']:<16s} {r['imgsz']:>5d} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
              f"{r['fps']:>6.1f} "
              f"{r['all_vehicles']['precision_50']:>6.4f} "
              f"{r['all_vehicles']['recall_50']:>6.4f} "
              f"{r['counting']['MAE']:>6.2f}")

    # ── Winner ──
    best_model = max(all_results, key=lambda r: r["all_vehicles"]["AP50"])
    print(f"\n  🏆 Best overall (vehicle AP@50): {best_model['model']} "
          f"({best_model['all_vehicles']['AP50']:.4f})")

    print(f"\n  Report saved to: {report_path}")
    print(f"  Done! ✅")


if __name__ == "__main__":
    main()
