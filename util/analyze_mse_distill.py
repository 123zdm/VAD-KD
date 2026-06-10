"""
MSE-only distillation analysis for AED-MAE.

When Stage-2 uses plain MSE on masked patch predictions, the test-time signal is:
  score = alpha * teacher_recon + beta * |teacher - student|

This script answers mechanism questions (NOT loss-family comparisons):
  1) Which score component separates normal vs abnormal best?
  2) On normal frames, how tight is the teacher-student gap after MSE distillation?
  3) On abnormal frames, does the gap grow relative to teacher recon?
  4) Spatially, does the TS-diff heatmap overlap GT anomaly regions?
  5) Does distillation only on masked patches explain gap behaviour (masked vs visible)?

Example:
  python util/analyze_mse_distill.py \\
    --teacher_checkpoint ../aed-mae_search/output/avenue/checkpoint-best.pth \\
    --student_checkpoint output/avenue/r0_mse_skip/checkpoint-best-student.pth \\
    --output_dir output/avenue/viz_mse_distill_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8


@dataclass
class FrameRecord:
    label: int
    video: str
    teacher_recon: float
    ts_gap: float
    fused_score: float
    gap_masked: float
    gap_visible: float
    distill_mse_masked: float


def build_model(args):
    return (
        mae_cvt_patch16(
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss,
            ts_loss_type="mse",
        ).float()
        if args.dataset == "avenue"
        else mae_cvt_patch8(
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss,
            ts_loss_type="mse",
        ).float()
    )


def load_student(args, teacher_ckpt: str, student_ckpt: str, device: torch.device):
    model = build_model(args).to(device)
    teacher = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"]
    student = torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"]
    for key in student:
        if "student" in key:
            teacher[key] = student[key]
    model.load_state_dict(teacher, strict=False)
    model.eval()
    model.train_TS = True
    return model


@torch.no_grad()
def extract_frame(model, samples, grads, targets, mask_ratio, args) -> FrameRecord:
    latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
    pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

    target_img = targets[:, :3]
    pred_teacher_img = model.unpatchify(pred_teacher)[:, :3]
    pred_stud_img = model.unpatchify(pred_stud)[:, :3]

    teacher_recon_map = ((target_img - pred_teacher_img) ** 2).mean(1)
    ts_gap_map = ((pred_teacher_img - pred_stud_img) ** 2).mean(1)

    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    fused_map = w_teacher * teacher_recon_map + w_ts * ts_gap_map

    # patch-level gap on masked vs visible tokens (distill only happens on masked)
    patch_gap = ((pred_teacher - pred_stud) ** 2).mean(-1)[0]
    m = mask[0].bool()
    gap_masked = float(patch_gap[m].mean().item()) if m.any() else 0.0
    gap_visible = float(patch_gap[~m].mean().item()) if (~m).any() else 0.0
    distill_mse_masked = float(patch_gap[m].mean().item()) if m.any() else 0.0

    return FrameRecord(
        label=int(targets.shape[0]),  # placeholder, overwritten by caller
        video="",
        teacher_recon=float(teacher_recon_map.amax().item()),
        ts_gap=float(ts_gap_map.amax().item()),
        fused_score=float(fused_map.amax().item()),
        gap_masked=gap_masked,
        gap_visible=gap_visible,
        distill_mse_masked=distill_mse_masked,
    )


@torch.no_grad()
def extract_frame_fixed(
    model, samples, grads, targets, label, video, mask_ratio, args
) -> Tuple[FrameRecord, np.ndarray, np.ndarray]:
    latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
    pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

    target_img = targets[:, :3]
    pred_teacher_img = model.unpatchify(pred_teacher)[:, :3]
    pred_stud_img = model.unpatchify(pred_stud)[:, :3]

    teacher_recon_map = ((target_img - pred_teacher_img) ** 2).mean(1)[0].cpu().numpy()
    ts_gap_map = ((pred_teacher_img - pred_stud_img) ** 2).mean(1)[0].cpu().numpy()

    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    fused_map = (
        w_teacher * ((target_img - pred_teacher_img) ** 2).mean(1)
        + w_ts * ((pred_teacher_img - pred_stud_img) ** 2).mean(1)
    )[0].cpu().numpy()

    patch_gap = ((pred_teacher - pred_stud) ** 2).mean(-1)[0]
    m = mask[0].bool()
    gap_masked = float(patch_gap[m].mean().item()) if m.any() else 0.0
    gap_visible = float(patch_gap[~m].mean().item()) if (~m).any() else 0.0

    rec = FrameRecord(
        label=int(label),
        video=str(video),
        teacher_recon=float(teacher_recon_map.max()),
        ts_gap=float(ts_gap_map.max()),
        fused_score=float(fused_map.max()),
        gap_masked=gap_masked,
        gap_visible=gap_visible,
        distill_mse_masked=gap_masked,
    )
    return rec, teacher_recon_map, ts_gap_map


def component_aucs(records: List[FrameRecord]) -> Dict[str, float]:
    y = np.array([r.label for r in records])
    if len(np.unique(y)) < 2:
        return {}
    out = {}
    for key in ["teacher_recon", "ts_gap", "fused_score", "gap_masked", "gap_visible"]:
        s = np.array([getattr(r, key) for r in records])
        out[key] = float(roc_auc_score(y, s))
    return out


def plot_component_separation(records: List[FrameRecord], out_path: Path):
    """Normal vs abnormal distributions for each score component."""
    metrics = [
        ("teacher_recon", "Teacher recon (max pixel)"),
        ("ts_gap", "TS gap (max pixel)"),
        ("fused_score", "Official fused (0.4T + 0.3TS)"),
        ("gap_masked", "Patch gap — masked tokens"),
        ("gap_visible", "Patch gap — visible tokens"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        normal = [getattr(r, key) for r in records if r.label == 0]
        abnormal = [getattr(r, key) for r in records if r.label == 1]
        if normal:
            ax.hist(normal, bins=35, density=True, histtype="step", linewidth=1.5, label="normal")
        if abnormal:
            ax.hist(abnormal, bins=35, density=True, histtype="step", linewidth=1.5,
                    linestyle="--", label="abnormal")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("MSE distillation: which signal separates normal vs abnormal?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gap_ratio(records: List[FrameRecord], out_path: Path):
    """On each frame: ts_gap / (teacher_recon + eps). Abnormal should have higher ratio if gap is the key."""
    ratios = []
    labels = []
    for r in records:
        ratios.append(r.ts_gap / (r.teacher_recon + 1e-6))
        labels.append(r.label)
    ratios = np.array(ratios)
    labels = np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, name, ls in [(0, "normal", "-"), (1, "abnormal", "--")]:
        vals = ratios[labels == lab]
        if len(vals):
            ax.hist(vals, bins=40, density=True, histtype="step", linewidth=1.6, linestyle=ls, label=name)
    ax.set_xlabel("TS_gap / teacher_recon")
    ax.set_ylabel("density")
    ax.set_title("Relative gap magnitude (abnormal → gap dominates over teacher recon?)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    if len(np.unique(labels)) == 2:
        return float(roc_auc_score(labels, ratios))
    return 0.0


def plot_masked_vs_visible(records: List[FrameRecord], out_path: Path):
    """Distillation targets masked patches only — is gap larger on masked tokens?"""
    masked = np.array([r.gap_masked for r in records])
    visible = np.array([r.gap_visible for r in records])
    labels = np.array([r.label for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, subset, title in [
        (axes[0], labels == 0, "Normal frames"),
        (axes[1], labels == 1, "Abnormal frames"),
    ]:
        if subset.any():
            ax.scatter(masked[subset], visible[subset], s=6, alpha=0.35)
            lim = max(masked[subset].max(), visible[subset].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("gap on masked patches")
        ax.set_ylabel("gap on visible patches")
        ax.set_title(title)
    fig.suptitle("Masked vs visible patch gap (MSE distill only on masked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_component_auc_bar(aucs: Dict[str, float], out_path: Path):
    order = ["teacher_recon", "ts_gap", "gap_masked", "gap_visible", "fused_score"]
    labels = [k for k in order if k in aucs]
    vals = [aucs[k] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e67e22" if k == "ts_gap" else "#3498db" for k in labels]
    ax.bar(labels, vals, color=colors, alpha=0.8)
    ax.set_ylabel("frame-level AUC (%)")
    ax.set_title("Per-component discriminability (unsmoothed, MSE student)")
    ax.set_ylim(0.5, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.1f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_spatial_example(
    teacher_map: np.ndarray,
    ts_map: np.ndarray,
    gt_mask: Optional[np.ndarray],
    out_path: Path,
    title: str,
):
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 3.5))
    panels = [
        (teacher_map, "Teacher recon"),
        (ts_map, "TS gap"),
    ]
    if gt_mask is not None:
        panels.append((gt_mask.astype(float), "GT mask"))
    for ax, (arr, t) in zip(axes, panels):
        im = ax.imshow(arr, cmap="hot")
        ax.set_title(t)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MSE-only AED-MAE distillation mechanism analysis")
    parser.add_argument("--dataset", default="avenue", choices=["avenue", "shanghai"])
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=800)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = get_configs_avenue() if args_cli.dataset == "avenue" else get_configs_shanghai()
    mask_ratio = args_cli.mask_ratio if args_cli.mask_ratio is not None else args.mask_ratio
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AbnormalDatasetGradientsTest(args)
    model = load_student(args, args_cli.teacher_checkpoint, args_cli.student_checkpoint, device)

    rng = np.random.default_rng(args_cli.seed)
    indices = np.arange(len(dataset))
    if args_cli.max_frames < len(indices):
        indices = rng.choice(indices, size=args_cli.max_frames, replace=False)

    records: List[FrameRecord] = []
    print(f"Extracting {len(indices)} frames ...")
    for i, idx in enumerate(indices):
        samples, grads, targets, label, video, _ = dataset[int(idx)]
        samples = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets = torch.from_numpy(targets).unsqueeze(0).to(device)
        rec, _, _ = extract_frame_fixed(model, samples, grads, targets, label, video, mask_ratio, args)
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(indices)}")

    aucs = component_aucs(records)
    ratio_auc = plot_gap_ratio(records, out_dir / "fig_gap_ratio.png")

    plot_component_separation(records, out_dir / "fig_component_separation.png")
    plot_masked_vs_visible(records, out_dir / "fig_masked_vs_visible.png")
    plot_component_auc_bar(aucs, out_dir / "fig_component_auc.png")

    # pick one abnormal + one normal frame for spatial example
    ab_idx = next((i for i, r in enumerate(records) if r.label == 1), None)
    nm_idx = next((i for i, r in enumerate(records) if r.label == 0), None)
    for tag, rec_idx in [("abnormal", ab_idx), ("normal", nm_idx)]:
        if rec_idx is None:
            continue
        idx = int(indices[rec_idx])
        samples, grads, targets, label, video, _ = dataset[idx]
        samples = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets = torch.from_numpy(targets).unsqueeze(0).to(device)
        _, t_map, g_map = extract_frame_fixed(
            model, samples, grads, targets, label, video, mask_ratio, args
        )
        gt = targets[0, 3].cpu().numpy() if targets.shape[1] > 3 else None
        plot_spatial_example(
            t_map, g_map, gt,
            out_dir / f"fig_spatial_{tag}_video{records[rec_idx].video}.png",
            f"Video {records[rec_idx].video} ({tag})",
        )

    summary = {
        "n_frames": len(records),
        "component_auc": aucs,
        "gap_ratio_auc": ratio_auc,
        "normal_gap_masked_mean": float(np.mean([r.gap_masked for r in records if r.label == 0])),
        "abnormal_gap_masked_mean": float(np.mean([r.gap_masked for r in records if r.label == 1])),
        "normal_gap_visible_mean": float(np.mean([r.gap_visible for r in records if r.label == 0])),
        "abnormal_gap_visible_mean": float(np.mean([r.gap_visible for r in records if r.label == 1])),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f)

    print("Component AUC (unsmoothed):")
    for k, v in sorted(aucs.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v*100:.2f}%")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
