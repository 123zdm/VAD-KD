"""
Paper Figure 4 style anomaly score curve for AED-MAE (CVPR 2024).

Official scoring: patch L2 TS scores → 0.4·teacher + 0.3·ts → filt()

Example:
  python util/visualize_paper_fig4.py \\
    --student_checkpoint output/avenue/r0_mse_skip/checkpoint-best-student.pth \\
    --video_id 04 \\
    --output_dir output/avenue/viz_paper_fig4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util.abnormal_utils import filt

def build_model(args):
    kwargs = dict(
        norm_pix_loss=args.norm_pix_loss,
        img_size=args.input_size,
        use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
        abnormal_score_func=args.abnormal_score_func,
        masking_method=args.masking_method,
        grad_weighted_loss=args.grad_weighted_rec_loss,
        ts_loss_type=args.ts_loss_type,
        bw2_eps=args.bw2_eps,
        ts_bw2_alpha=args.ts_bw2_alpha,
    )
    if args.dataset == "avenue":
        return mae_cvt_patch16(**kwargs).float()
    return mae_cvt_patch8(**kwargs).float()


def remap_checkpoint_keys(state: dict) -> dict:
    return dict(state)


def load_model(args, student_ckpt: str, device: torch.device):
    model = build_model(args).to(device)
    state = remap_checkpoint_keys(
        torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"]
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:6]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:6]}")
    model.eval()
    model.train_TS = True
    if args.dataset == "avenue":
        model.abnormal_score_func_TS = "L2"
    return model


def indices_for_video(dataset, video_id: str) -> List[int]:
    return [
        i for i in range(len(dataset))
        if dataset.data[i].split("/")[-2] == str(video_id)
    ]


def fuse_ts_teacher_scores(ts_score, teacher_score, args) -> torch.Tensor:
    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    return w_teacher * teacher_score + w_ts * ts_score


@torch.no_grad()
def forward_official_frame_score(model, samples, grads, targets, mask_ratio, args) -> float:
    """Match engine_train.test_one_epoch scoring path."""
    _, _, _, recon_error = model(
        samples, targets=targets, grad_mask=grads, mask_ratio=mask_ratio
    )
    if isinstance(recon_error, (list, tuple)):
        fused = fuse_ts_teacher_scores(recon_error[0], recon_error[1], args)
        return float(fused[0].item())
    return float(recon_error[0].item())


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def apply_temporal_smooth(scores: np.ndarray, args) -> np.ndarray:
    return filt(
        scores,
        range=getattr(args, "smooth_range", 38 if args.dataset == "avenue" else 900),
        mu=getattr(args, "smooth_mu", 11 if args.dataset == "avenue" else 282),
    )


def gt_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(labels.astype(bool)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def video_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    lbl = np.array([0] + list(labels) + [1])
    pred = np.array([0] + list(scores) + [1])
    return float(metrics.auc(*metrics.roc_curve(lbl, pred)[:2]))


def collect_video_scores(
    model,
    dataset,
    video_id: str,
    device: torch.device,
    mask_ratio: float,
    args,
) -> Dict:
    indices = indices_for_video(dataset, video_id)
    if not indices:
        raise ValueError(f"Video {video_id} not found")

    labels: List[int] = []
    raw_scores_list: List[float] = []
    print(f"  forward {len(indices)} frames (official) ...", flush=True)
    for j, idx in enumerate(indices):
        if j == 0 or (j + 1) % 100 == 0 or j + 1 == len(indices):
            print(f"    {j + 1}/{len(indices)}", flush=True)
        samples, grads, targets, label, _, _ = dataset[idx]
        samples_t = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads_t = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets_t = torch.from_numpy(targets).unsqueeze(0).to(device)
        raw_scores_list.append(
            forward_official_frame_score(model, samples_t, grads_t, targets_t, mask_ratio, args)
        )
        labels.append(int(label))
    raw_scores = np.asarray(raw_scores_list, dtype=float)

    labels_arr = np.asarray(labels, dtype=int)
    temporal_smooth = apply_temporal_smooth(raw_scores, args)
    scores_norm = minmax_normalize(temporal_smooth)
    auc_raw = video_auc(raw_scores, labels_arr)
    auc_smooth = video_auc(temporal_smooth, labels_arr)

    return {
        "labels": labels_arr,
        "raw_scores": raw_scores,
        "temporal_smooth": temporal_smooth,
        "scores_norm": scores_norm,
        "video_auc_raw": auc_raw,
        "video_auc": auc_smooth,
    }


def save_score_curve(video_id: str, data: Dict, out_path: Path):
    scores = data["scores_norm"]
    labels = data["labels"]
    auc = data["video_auc"]
    n = len(scores)
    frames = np.arange(n)

    fig, (ax, ax_gt) = plt.subplots(
        2, 1, figsize=(12, 4.2), sharex=True, gridspec_kw={"height_ratios": [3.5, 1]}
    )

    for i, (s, e) in enumerate(gt_segments(labels)):
        ax.axvspan(s, e, color="#FFB3BA", alpha=0.45, label="Ground-truth" if i == 0 else None)

    ax.plot(
        frames,
        scores,
        color="#2E7D32",
        linewidth=2.2,
        label=f"Official — AUC = {auc * 100:.2f}%",
    )
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Video {video_id} — Avenue")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    ax_gt.fill_between(frames, 0, labels.astype(int), step="mid", color="#e74c3c", alpha=0.85)
    ax_gt.set_ylim(-0.05, 1.05)
    ax_gt.set_yticks([0, 1])
    ax_gt.set_yticklabels(["normal", "abnormal"], fontsize=8)
    ax_gt.set_ylabel("GT", fontsize=9)
    n_ab = int(labels.sum())
    ax_gt.set_xlabel(f"Frame index  (GT abnormal: {n_ab}/{n} frames)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="AED-MAE paper-style anomaly score curve")
    parser.add_argument("--dataset", default="avenue", choices=["avenue", "shanghai"])
    parser.add_argument("--teacher_checkpoint", default="", help="unused; kept for CLI compat")
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--video_id", default="04")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = get_configs_avenue() if args_cli.dataset == "avenue" else get_configs_shanghai()
    mask_ratio = args_cli.mask_ratio if args_cli.mask_ratio is not None else args.mask_ratio
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model (official fusion) ...")
    model = load_model(args, args_cli.student_checkpoint, device)

    dataset = AbnormalDatasetGradientsTest(args)
    print(f"Collecting scores for video {args_cli.video_id} ...")
    data = collect_video_scores(
        model,
        dataset,
        args_cli.video_id,
        device,
        mask_ratio,
        args,
    )

    fig_path = out_dir / f"score_official_video_{args_cli.video_id}.png"
    save_score_curve(args_cli.video_id, data, fig_path)

    np.savez(
        out_dir / f"score_official_video_{args_cli.video_id}.npz",
        labels=data["labels"],
        raw_scores=data["raw_scores"],
        temporal_smooth=data["temporal_smooth"],
        scores_norm=data["scores_norm"],
        video_auc_raw=data["video_auc_raw"],
        video_auc=data["video_auc"],
    )

    summary = {
        "video_id": args_cli.video_id,
        "score_mode": "official",
        "video_auc_raw": data["video_auc_raw"],
        "video_auc_smooth": data["video_auc"],
        "n_frames": int(len(data["labels"])),
        "n_abnormal": int(data["labels"].sum()),
        "figure": str(fig_path),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Video AUC (raw):    {data['video_auc_raw'] * 100:.2f}%")
    print(f"Video AUC (smooth): {data['video_auc'] * 100:.2f}%")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
