#!/usr/bin/env python3
"""
YOLO Model Evaluation on UAVDT-Preview Dataset
================================================
Evaluates all YOLO models in test/model/ against UAVDT-preview images
(train + test splits combined), using existing Supervisely JSON labels
as ground truth.

Metrics computed:
  - Per-class AP@50, AP@50:95, Precision, Recall
  - Overall mAP@50, mAP@50:95
  - Inference speed (FPS)
  - Vehicle counting accuracy (MAE, RMSE per image)
  - Confusion-style analysis

Usage:
    conda activate cvenv
    python evaluate_models.py [--conf 0.25] [--iou 0.5] [--imgsz 960]
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

# ─── PATHS ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "UAVDT-preview"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── CLASS MAPPING ─────────────────────────────────────────────────────
# UAVDT ground-truth classes from meta.json
UAVDT_CLASSES = ["bus", "car", "truck", "vehicle"]

# VisDrone classes (what the YOLO models were trained on)
VISDRONE_CLASSES = [
    "pedestrian",   # 0
    "people",       # 1
    "bicycle",      # 2
    "car",          # 3
    "van",          # 4
    "truck",        # 5
    "tricycle",     # 6
    "awning-tricycle",  # 7
    "bus",          # 8
    "motor",        # 9
]

# Map VisDrone class indices → unified evaluation class name
# We merge car+van → car, and treat "vehicle" labels as a generic match
VISDRONE_TO_EVAL = {
    3: "car",       # car
    4: "car",       # van → car
    5: "truck",     # truck
    8: "bus",       # bus
}

# For the "vehicle" GT class (found in test split), any of these pred classes match
VEHICLE_PRED_CLASSES = {"car", "truck", "bus"}


# ─── ANNOTATION PARSING ───────────────────────────────────────────────

def parse_supervisely_json(json_path: Path) -> list[dict]:
    """
    Parse a Supervisely JSON annotation file.
    Returns list of dicts: {class_name, bbox: [x1,y1,x2,y2]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = []
    for obj in data.get("objects", []):
        cls = obj.get("classTitle", "").lower()
        if cls not in UAVDT_CLASSES:
            continue
        pts = obj["points"]["exterior"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        annotations.append({
            "class_name": cls,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        })
    return annotations


def collect_dataset() -> list[dict]:
    """
    Collect all images from train/ and test/ splits with their ground truth.
    Returns list of {img_path, ann_path, split, gt_boxes: [...]}
    """
    dataset = []
    for split in ["train", "test"]:
        img_dir = DATASET_DIR / split / "img"
        ann_dir = DATASET_DIR / split / "ann"
        if not img_dir.exists():
            print(f"  ⚠ Split '{split}' not found, skipping...")
            continue

        img_files = sorted(img_dir.glob("*.jpg"))
        for img_path in img_files:
            ann_path = ann_dir / f"{img_path.name}.json"
            gt_boxes = []
            if ann_path.exists():
                gt_boxes = parse_supervisely_json(ann_path)

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

def map_prediction_class(yolo_cls_id: int, model) -> str | None:
    """Map a YOLO prediction class ID to our eval class name."""
    cls_name = model.names.get(yolo_cls_id, "")
    
    # Direct VisDrone mapping
    if yolo_cls_id in VISDRONE_TO_EVAL:
        return VISDRONE_TO_EVAL[yolo_cls_id]
    
    # Fallback: try matching by name
    name_lower = cls_name.lower()
    if name_lower in ["car", "van"]:
        return "car"
    elif name_lower == "truck":
        return "truck"
    elif name_lower == "bus":
        return "bus"
    
    return None  # Not a vehicle class we care about


def run_evaluation(model_name: str, model_path: str, dataset: list[dict],
                   conf: float, iou_thresh: float, imgsz: int) -> dict:
    """Run full evaluation for one model."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Model:      {model_path}")
    print(f"  Images:     {len(dataset)}")
    print(f"  Conf:       {conf}  |  IoU:  {iou_thresh}  |  ImgSz: {imgsz}")
    print(f"{'='*70}")

    model = YOLO(model_path)
    
    # Print model class names for reference
    print(f"  Model classes: {model.names}")

    # Storage for per-class evaluations
    # Classes to evaluate: bus, car, truck, vehicle (vehicle = any vehicle)
    eval_classes = ["bus", "car", "truck"]
    gt_by_class = {c: [] for c in eval_classes}        # (img_idx, bbox)
    det_by_class = {c: [] for c in eval_classes}       # (img_idx, conf, bbox)

    # For "vehicle" GT labels → match against any vehicle prediction
    gt_vehicle = []     # (img_idx, bbox)
    det_vehicle = []    # (img_idx, conf, bbox) — all vehicle-type preds

    # Counting metrics
    per_image_gt_counts = []
    per_image_det_counts = []

    # Per-split metrics
    split_stats = {"train": {"n": 0, "gt": 0, "det": 0},
                   "test": {"n": 0, "gt": 0, "det": 0}}

    total_inference_time = 0.0
    n_processed = 0

    for img_idx, sample in enumerate(dataset):
        img_path = sample["img_path"]
        gt_boxes = sample["gt_boxes"]
        split = sample["split"]

        # ── Run inference ──
        t0 = time.time()
        results = model.predict(
            img_path,
            conf=conf,
            iou=iou_thresh,
            imgsz=imgsz,
            verbose=False,
            device="mps",  # Use Apple Silicon GPU
        )
        t1 = time.time()
        total_inference_time += (t1 - t0)
        n_processed += 1

        # ── Parse predictions ──
        preds = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    eval_cls = map_prediction_class(cls_id, model)
                    if eval_cls:
                        preds.append({
                            "class_name": eval_cls,
                            "confidence": confidence,
                            "bbox": xyxy,
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
                # Also add to the "all vehicles" bucket
                gt_vehicle.append((img_idx, gt["bbox"]))

        for pred in preds:
            pred_cls = pred["class_name"]
            if pred_cls in eval_classes:
                det_by_class[pred_cls].append(
                    (img_idx, pred["confidence"], pred["bbox"]))
            # All vehicle predictions go to the vehicle bucket
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
    print(f"  └──────────────────────────────────────────────┘")

    return result


def generate_comparison_table(all_results: list[dict]) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append("# YOLO Model Comparison on UAVDT-Preview")
    lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: UAVDT-Preview (train + test splits combined)")
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
        description="Evaluate YOLO models on UAVDT-Preview dataset")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Inference image size (default: 960)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   YOLO Model Evaluation on UAVDT-Preview Dataset    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Discover models ──
    models = []
    if MODEL_DIR.exists():
        for model_dir in sorted(MODEL_DIR.iterdir()):
            if model_dir.is_dir():
                best_pt = model_dir / "best.pt"
                if best_pt.exists():
                    models.append({
                        "name": model_dir.name,
                        "path": str(best_pt),
                    })

    if not models:
        print("❌ No models found in", MODEL_DIR)
        sys.exit(1)

    print(f"\n  Found {len(models)} model(s):")
    for m in models:
        print(f"    • {m['name']}: {m['path']}")

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
            imgsz=args.imgsz,
        )
        all_results.append(result)

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Individual model results
    for result in all_results:
        out_path = RESULTS_DIR / f"{result['model']}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

    # Comparison JSON
    comparison_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {comparison_path}")

    # Markdown report
    report = generate_comparison_table(all_results)
    report_path = RESULTS_DIR / f"report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # ── Final comparison ──
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Model':<16s} {'mAP@50':>8s} {'mAP@50:95':>10s} "
          f"{'FPS':>6s} {'P':>6s} {'R':>6s} {'MAE':>6s}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in all_results:
        print(f"  {r['model']:<16s} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
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
