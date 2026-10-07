import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)

def get_optimal_f1(groundtruth, probabilities, return_threshold=False):
    prec, rec, threshold = precision_recall_curve(groundtruth, probabilities)
    f1_values = 2 * (prec * rec) / (prec + rec + 1e-10)
    argmax_f1 = np.nanargmax(f1_values)
    max_f1 = np.nanmax(f1_values)

    if return_threshold:
        return max_f1, threshold[argmax_f1]
    else:
        return max_f1

def get_max_precision_above_recall(groundtruth, probabilities, value, return_threshold=False):
    if value > 1:
        raise ValueError(f"Cannot attain a recall of {value}")
    prec, rec, threshold = precision_recall_curve(groundtruth, probabilities)
    max_prec_above_rec = max(p for p, r in zip(prec, rec) if r >= value)

    if return_threshold:
        index = list(prec).index(max_prec_above_rec)
        return max_prec_above_rec, threshold[index - 1]
    else:
        return max_prec_above_rec

def get_optimal_segmentation_f1(groundtruth, probabilities, return_threshold=False):
    prec, rec, thresholds = precision_recall_curve(groundtruth, probabilities)
    f1_values = 2 * (prec * rec) / (prec + rec + 1e-10)
    argmax_f1 = np.nanargmax(f1_values)
    max_f1 = np.nanmax(f1_values)

    if return_threshold:
        return max_f1, thresholds[argmax_f1]
    else:
        return max_f1


def get_segmentation_metrics(probs, labels, threshold=None):

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    labels = labels.reshape(-1)
    probs = probs.reshape(-1)

    iou_preds = torch.sigmoid(probs)

    if threshold is None:
        _, threshold = get_optimal_f1(labels, probs, return_threshold=True)

    preds = (probs > threshold).astype(int)
    labels = labels.astype(int)
    # Compute Intersection and Union
    intersection_0 = np.sum(preds * labels)  # Sum of overlapping pixels
    union_0 = np.sum(preds) + np.sum(labels) - intersection_0  # Total unique pixels in either mask
    # Compute IoU (Jaccard Index)
    iou_0 = intersection_0 / (union_0 + 1e-7)  # Avoid division by zero


    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    intersection = np.logical_and(labels, preds).sum()
    union = np.logical_or(labels, preds).sum()
    iou = intersection / union if union != 0 else 0

    dice = 2 * intersection / (labels.sum() + preds.sum() + 1e-10)

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0
    auprc = average_precision_score(labels, probs)

    prevalence = np.mean(labels)

    return {
        'threshold': threshold,
        'prevalence': prevalence,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'iou': iou,
        'dice': dice,
        'iou_0': iou_0,
    }

def test_simple_iou():
    # Create a simple 4x4 ground truth mask and predicted mask
    ground_truth = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    predicted = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=float)
    
    ground_truth = ground_truth[np.newaxis, np.newaxis, :, :]
    predicted = predicted[np.newaxis, np.newaxis, :, :]

    b, c, h, w = ground_truth.shape
    print(b, c, h, w)
    ground_truth_flat = ground_truth.reshape(b, c * h * w)
    predicted_flat = predicted.reshape(b, c * h * w)

    ground_truth_flat = ground_truth_flat.reshape(-1)
    predicted_flat = predicted_flat.reshape(-1)

    metrics = get_segmentation_metrics(predicted_flat, ground_truth_flat, threshold=0.5)
    print(metrics['iou'])


if __name__ == "__main__":
    test_simple_iou()
