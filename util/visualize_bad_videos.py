"""
Diagnose why specific Avenue videos score poorly: FN/FP frames + pixel heatmaps.

Outputs under --output_dir:
  curves/video_{id}_diagnosis.png     fused / teacher / ts_gap vs GT
  curves/video_{id}_errors.png        FN & FP regions marked
  pixel/video_{id}_{FN|FP}_{frame}.png  4-panel heatmaps for worst frames
  diagnosis_report.md

Example:
  python util/visualize_bad_videos.py \\
    --teacher_checkpoint ../aed-mae_search/output/avenue_cls_head/checkpoint-best.pth \\
    --student_checkpoint ../aed-mae_search/output/avenue_cls_head/checkpoint-best-student.pth \\
    --output_dir output/avenue/bad_video_diagnosis \\
    --video_ids 17,20,16,01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue
from data.test_dataset import AbnormalDatasetGradientsTest
from inference import fuse_ts_teacher_scores
from model.model_factory import mae_cvt_patch16
from util.abnormal_utils import filt
from util.visualize_paper_fig4 import remap_checkpoint_keys
from util.visualize_vad_analysis import (
    add_gt_shading,
    channel_mse,
    forward_pixel_maps,
    gt_contiguous_segments,
    indices_for_video,
    normalize_map,
    plot_gt_strip,
    save_pixel_4panel,
    tensor_to_display_rgb,
)


def load_model(args, teacher_ckpt: str, student_ckpt: str, device: torch.device):
    model = mae_cvt_patch16(
        norm_pix_loss=args.norm_pix_loss,
        img_size=args.input_size,
        use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
        abnormal_score_func=args.abnormal_score_func,
        masking_method=args.masking_method,
        grad_weighted_loss=args.grad_weighted_rec_loss,
        ts_loss_type="mse",
    ).float().to(device)

    teacher = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"]
    student = remap_checkpoint_keys(
        torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"]
    )
    for key in student:
        if "student" in key:
            teacher[key] = student[key]
    model.load_state_dict(remap_checkpoint_keys(teacher), strict=False)
    model.eval()
    model.train_TS = True
    model.abnormal_score_func_TS = "L2"
    return model


@torch.no_grad()
def frame_scores(model, samples, grads, targets, mask_ratio, args) -> Dict[str, float]:
    latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
    pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)
    ts_scores, teacher_scores = model.abnormal_score_TS(
        targets, pred_stud, pred_teacher, mask, grads
    )
    ts_gap = float(ts_scores[0].item())
    teacher_recon = float(teacher_scores[0].item())
    fused = float(
        fuse_ts_teacher_scores(
            torch.tensor([ts_gap]), torch.tensor([teacher_recon]), args
        )[0].item()
    )
    return {"fused": fused, "teacher": teacher_recon, "ts_gap": ts_gap}


def collect_video(model, dataset, device, mask_ratio, args, video_id: str) -> Dict:
    indices = indices_for_video(dataset, video_id)
    scores = {"fused": [], "teacher": [], "ts_gap": [], "labels": [], "paths": []}
    for i in indices:
        samples, grads, targets, label, _, path = dataset[i]
        samples_t = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads_t = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets_t = torch.from_numpy(targets).unsqueeze(0).to(device)
        sc = frame_scores(model, samples_t, grads_t, targets_t, mask_ratio, args)
        for k in ("fused", "teacher", "ts_gap"):
            scores[k].append(sc[k])
        scores["labels"].append(int(label))
        scores["paths"].append(path)
    for k in ("fused", "teacher", "ts_gap", "labels"):
        scores[k] = np.array(scores[k], dtype=float if k != "labels" else int)
    scores["fused_smooth"] = filt(scores["fused"], range=args.smooth_range, mu=args.smooth_mu)
    scores["teacher_smooth"] = filt(scores["teacher"], range=args.smooth_range, mu=args.smooth_mu)
    scores["ts_smooth"] = filt(scores["ts_gap"], range=args.smooth_range, mu=args.smooth_mu)
    return scores


def video_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    lbl = np.array([0] + list(labels.astype(int)) + [1])
    pred = np.array([0] + list(preds) + [1])
    fpr, tpr, _ = metrics.roc_curve(lbl, pred)
    return float(metrics.auc(fpr, tpr))


def optimal_threshold(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, float]:
    """Youden J threshold on padded ROC (same as per-video AUC eval)."""
    lbl = np.array([0] + list(labels.astype(int)) + [1])
    pred = np.array([0] + list(preds) + [1])
    fpr, tpr, thr = metrics.roc_curve(lbl, pred)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(metrics.auc(fpr, tpr))


def classify_errors(labels: np.ndarray, preds: np.ndarray, threshold: float) -> Dict[str, np.ndarray]:
    ab = labels.astype(bool)
    pred_pos = preds >= threshold
    return {
        "fn": ab & ~pred_pos,
        "fp": ~ab & pred_pos,
        "tp": ab & pred_pos,
        "tn": ~ab & ~pred_pos,
    }


def plot_diagnosis_curves(scores: Dict, video_id: str, auc_fused: float, out_path: Path):
    labels = scores["labels"]
    frames = np.arange(len(labels))
    n_ab = int(labels.sum())

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2.5, 1]})
    streams = [
        ("fused_smooth", "Official fused (0.4·T + 0.3·TS)", "tab:red"),
        ("teacher_smooth", "Teacher recon", "tab:blue"),
        ("ts_smooth", "TS gap", "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes[:3], streams):
        add_gt_shading(ax, labels, frames)
        ax.plot(frames, scores[key], color=color, linewidth=1.8, label=title)
        ax.set_ylabel("Score")
        ax.set_title(f"Video {video_id} — {title}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"Video {video_id} — Official fused  (AUC={auc_fused*100:.1f}%, abnormal={n_ab}/{len(labels)})")
    plot_gt_strip(axes[3], labels, frames)
    axes[3].set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_markers(scores: Dict, errors: Dict, threshold: float, video_id: str, out_path: Path):
    labels = scores["labels"]
    preds = scores["fused_smooth"]
    frames = np.arange(len(labels))

    fig, (ax, ax_gt) = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    add_gt_shading(ax, labels, frames)
    ax.plot(frames, preds, color="tab:red", linewidth=1.8, label="fused (filt)", zorder=2)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold={threshold:.4f}")

    fn_idx = np.where(errors["fn"])[0]
    fp_idx = np.where(errors["fp"])[0]
    if len(fn_idx):
        ax.scatter(fn_idx, preds[fn_idx], c="gold", s=40, edgecolors="black", linewidths=0.5,
                   label=f"FN ({len(fn_idx)})", zorder=4)
    if len(fp_idx):
        ax.scatter(fp_idx, preds[fp_idx], c="cyan", s=30, edgecolors="black", linewidths=0.5,
                   label=f"FP ({len(fp_idx)})", zorder=4)

    ax.set_ylabel("Score")
    ax.set_title(
        f"Video {video_id} errors  "
        f"(FN={len(fn_idx)}, FP={len(fp_idx)}, TP={errors['tp'].sum()}, TN={errors['tn'].sum()})"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    plot_gt_strip(ax_gt, labels, frames)
    ax_gt.set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fn_fp_panel(
    model, dataset, device, mask_ratio, idx: int, tag: str, video_id: str,
    score: float, out_path: Path,
):
    samples, grads, targets, label, _, path = dataset[idx]
    samples_t = torch.from_numpy(samples).unsqueeze(0).to(device)
    grads_t = torch.from_numpy(grads).unsqueeze(0).to(device)
    targets_t = torch.from_numpy(targets).unsqueeze(0).to(device)
    maps = forward_pixel_maps(model, samples_t, grads_t, targets_t, mask_ratio)
    frame_name = Path(path).stem
    title = (
        f"Video {video_id} {tag} frame {frame_name}  "
        f"GT={int(label)} score={score:.4f}"
    )
    save_pixel_4panel(maps, out_path, title, "author")

    # extra overlay: GT abnormal strip isn't pixel-level; show recon vs ts side by side
    gt = tensor_to_display_rgb(maps["target_rgb"])
    recon = maps["recon_teacher"].cpu().numpy()
    ts = maps["ts_diff"].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt)
    axes[0].set_title("RGB frame")
    im1 = axes[1].imshow(recon, cmap="hot")
    axes[1].set_title(f"Teacher recon (max={recon.max():.3f})")
    im2 = axes[2].imshow(ts, cmap="hot")
    axes[2].set_title(f"TS gap (max={ts.max():.3f})")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem + "_detail.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnose_failure_mode(scores: Dict, errors: Dict, threshold: float) -> str:
    labels = scores["labels"]
    ab = labels.astype(bool)
    if not ab.any():
        return "无 GT 异常帧"

    fn_rate = errors["fn"].sum() / max(ab.sum(), 1)
    fp_rate = errors["fp"].sum() / max((~ab).sum(), 1)

    ab_teacher = scores["teacher_smooth"][ab].mean()
    nm_teacher = scores["teacher_smooth"][~ab].mean()
    ab_ts = scores["ts_smooth"][ab].mean()
    nm_ts = scores["ts_smooth"][~ab].mean()

    lines = []
    if fn_rate > 0.5:
        lines.append(f"**漏检严重**：{errors['fn'].sum()}/{ab.sum()} 异常帧低于阈值 ({fn_rate*100:.0f}%)")
    elif fn_rate > 0.2:
        lines.append(f"**部分漏检**：{errors['fn'].sum()}/{ab.sum()} 异常帧未检出")

    if fp_rate > 0.15:
        lines.append(f"**误报较多**：{errors['fp'].sum()}/{(~ab).sum()} 正常帧被标为异常 ({fp_rate*100:.0f}%)")

    if ab_teacher <= nm_teacher * 1.1:
        lines.append(
            f"Teacher recon 对异常/正常 **区分度弱** "
            f"(异常均值 {ab_teacher:.4f} vs 正常 {nm_teacher:.4f})"
        )
    else:
        lines.append(
            f"Teacher recon 有区分度 (异常 {ab_teacher:.4f} > 正常 {nm_teacher:.4f})，"
            "但融合后仍漏检 → 检查峰值是否被平滑抹平"
        )

    if ab_ts <= nm_ts:
        lines.append(
            f"TS gap **无帮助**：异常帧 gap ({ab_ts:.4f}) ≤ 正常 ({nm_ts:.4f})，蒸馏过拟合"
        )
    elif ab_ts < ab_teacher * 0.5:
        lines.append(
            f"TS gap 有微弱区分 (异常 {ab_ts:.4f} vs 正常 {nm_ts:.4f})，"
            "但被 teacher 主导融合淹没"
        )
    else:
        lines.append(f"TS gap 区分度尚可 (异常 {ab_ts:.4f} vs 正常 {nm_ts:.4f})")

    segs = gt_contiguous_segments(labels)
    missed_segs = []
    for s, e in segs:
        seg_fn = errors["fn"][s:e + 1].mean()
        if seg_fn > 0.5:
            missed_segs.append(f"{s}-{e}")
    if missed_segs:
        lines.append(f"整段漏检区间：帧 {', '.join(missed_segs)}")

    return "；".join(lines) if lines else "无明显单一模式"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_ids", default="17,20,16,01")
    parser.add_argument("--top_k_frames", type=int, default=3, help="FN/FP frames to visualize each")
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = get_configs_avenue()
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args_cli.output_dir)
    curve_dir = out_dir / "curves"
    pixel_dir = out_dir / "pixel"
    curve_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir.mkdir(parents=True, exist_ok=True)

    video_ids = [v.strip() for v in args_cli.video_ids.split(",") if v.strip()]
    dataset = AbnormalDatasetGradientsTest(args)
    model = load_model(args, args_cli.teacher_checkpoint, args_cli.student_checkpoint, device)

    report = {"videos": {}, "checkpoint": args_cli.student_checkpoint}
    md_lines = [
        "# 差视频诊断报告",
        "",
        f"Checkpoint: `{args_cli.student_checkpoint}`",
        "",
        "| 视频 | Fused AUC | Teacher AUC | TS AUC | FN | FP | 诊断 |",
        "|------|-----------|-------------|--------|----|----|------|",
    ]

    for vid in video_ids:
        print(f"Processing video {vid} ...")
        scores = collect_video(model, dataset, device, args.mask_ratio, args, vid)
        labels = scores["labels"]
        auc_fused = video_auc(labels, scores["fused_smooth"])
        auc_teacher = video_auc(labels, scores["teacher_smooth"])
        auc_ts = video_auc(labels, scores["ts_smooth"])
        threshold, _ = optimal_threshold(labels, scores["fused_smooth"])
        errors = classify_errors(labels, scores["fused_smooth"], threshold)
        diagnosis = diagnose_failure_mode(scores, errors, threshold)

        plot_diagnosis_curves(scores, vid, auc_fused, curve_dir / f"video_{vid}_diagnosis.png")
        plot_error_markers(scores, errors, threshold, vid, curve_dir / f"video_{vid}_errors.png")

        # worst FN: abnormal frames with lowest score
        fn_indices = np.where(errors["fn"])[0]
        fp_indices = np.where(errors["fp"])[0]
        fn_pick = fn_indices[np.argsort(scores["fused_smooth"][fn_indices])[:args_cli.top_k_frames]] if len(fn_indices) else []
        fp_pick = fp_indices[np.argsort(-scores["fused_smooth"][fp_indices])[:args_cli.top_k_frames]] if len(fp_indices) else []

        global_indices = indices_for_video(dataset, vid)
        for rank, local_i in enumerate(fn_pick):
            gi = global_indices[local_i]
            save_fn_fp_panel(
                model, dataset, device, args.mask_ratio, gi, "FN",
                vid, scores["fused_smooth"][local_i],
                pixel_dir / f"video_{vid}_FN_{rank}_{Path(scores['paths'][local_i]).stem}.png",
            )
        for rank, local_i in enumerate(fp_pick):
            gi = global_indices[local_i]
            save_fn_fp_panel(
                model, dataset, device, args.mask_ratio, gi, "FP",
                vid, scores["fused_smooth"][local_i],
                pixel_dir / f"video_{vid}_FP_{rank}_{Path(scores['paths'][local_i]).stem}.png",
            )

        segs = gt_contiguous_segments(labels)
        report["videos"][vid] = {
            "auc_fused": auc_fused,
            "auc_teacher": auc_teacher,
            "auc_ts": auc_ts,
            "threshold": threshold,
            "n_frames": int(len(labels)),
            "n_abnormal": int(labels.sum()),
            "fn": int(errors["fn"].sum()),
            "fp": int(errors["fp"].sum()),
            "tp": int(errors["tp"].sum()),
            "tn": int(errors["tn"].sum()),
            "gt_segments": [[int(s), int(e)] for s, e in segs],
            "diagnosis": diagnosis,
            "mean_score_abnormal": float(scores["fused_smooth"][labels.astype(bool)].mean()),
            "mean_score_normal": float(scores["fused_smooth"][~labels.astype(bool)].mean()),
            "fn_frames_visualized": [int(i) for i in fn_pick],
            "fp_frames_visualized": [int(i) for i in fp_pick],
        }
        md_lines.append(
            f"| {vid} | {auc_fused*100:.1f}% | {auc_teacher*100:.1f}% | {auc_ts*100:.1f}% | "
            f"{errors['fn'].sum()} | {errors['fp'].sum()} | {diagnosis} |"
        )

    md_lines += [
        "",
        "## 图表说明",
        "",
        "- `curves/video_XX_diagnosis.png`：三分量曲线 + GT 条",
        "- `curves/video_XX_errors.png`：FN（黄）/ FP（青）帧标注",
        "- `pixel/video_XX_FN/FP_*.png`：漏检/误报代表性帧的 teacher recon 与 TS gap 热力图",
        "",
        "## 如何读图",
        "",
        "1. **FN 帧**：GT 异常但分数低 → 看 teacher recon 是否在异常区域也偏低（重建太好）",
        "2. **FP 帧**：GT 正常但分数高 → 看是否背景运动/光照导致 recon 误差偏高",
        "3. **整段漏检**：GT 条有红色区间但曲线无峰 → 异常太弱或与训练分布不一致",
    ]

    with open(out_dir / "diagnosis_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    (out_dir / "diagnosis_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
