"""
Performance evaluator for multi-person localization.
Uses Hungarian matching to pair predictions with ground truth per frame,
then reports mean localization error, precision, recall, and F1.
"""
import argparse
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment


MATCH_THRESHOLD = 0.5  # metres — predictions farther than this are false positives


def load_jsonl(path: str) -> dict[int, list]:
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data[obj["frame"]] = obj["localizations"]
    return data


def hungarian_match(gt_points: list, pred_points: list, threshold: float):
    """
    Returns (matched_distances, n_tp, n_fp, n_fn).
    """
    if len(gt_points) == 0 and len(pred_points) == 0:
        return [], 0, 0, 0
    if len(gt_points) == 0:
        return [], 0, len(pred_points), 0
    if len(pred_points) == 0:
        return [], 0, 0, len(gt_points)

    gt = np.array(gt_points)   # (G, 2)
    pr = np.array(pred_points)  # (P, 2)

    # Cost matrix: pairwise euclidean distance
    cost = np.linalg.norm(gt[:, None, :] - pr[None, :, :], axis=-1)  # (G, P)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_dist = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        d = cost[r, c]
        if d <= threshold:
            matched_dist.append(d)
            tp += 1

    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    return matched_dist, tp, fp, fn


def evaluate(gt_path: str, pred_path: str, threshold: float = MATCH_THRESHOLD):
    gt_data = load_jsonl(gt_path)
    pred_data = load_jsonl(pred_path)

    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))

    total_tp = total_fp = total_fn = 0
    all_distances = []

    for frame in all_frames:
        gt_pts = gt_data.get(frame, [])
        pr_pts = pred_data.get(frame, [])
        dists, tp, fp, fn = hungarian_match(gt_pts, pr_pts, threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_distances.extend(dists)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_dist = float(np.mean(all_distances)) if all_distances else float("nan")

    return {
        "frames_evaluated": len(all_frames),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_error_m": mean_dist,
        "match_threshold_m": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate localization performance")
    parser.add_argument("--gt-path", required=True, help="Ground truth .jsonl path")
    parser.add_argument("--pred-path", required=True, help="Predicted .jsonl path")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"Match distance threshold in metres (default: {MATCH_THRESHOLD})")
    args = parser.parse_args()

    metrics = evaluate(args.gt_path, args.pred_path, args.threshold)

    print(f"Frames evaluated : {metrics['frames_evaluated']}")
    print(f"TP / FP / FN     : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1               : {metrics['f1']:.4f}")
    print(f"Mean loc. error  : {metrics['mean_localization_error_m']:.4f} m  (threshold={args.threshold} m)")


if __name__ == "__main__":
    main()
