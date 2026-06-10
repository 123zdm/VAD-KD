"""
Pixel-level heatmaps and per-video anomaly score curves for AED-MAE VAD analysis.

Inspired by hstforu-kd visualize_pixel_anomaly.py / visualize_score.py, adapted for
teacher-student MAE reconstruction (unpatchify + multi-stream errors).

Outputs:
  pixel/
    pixel_4panel_{video}_{frame}_{student}.png   GT | teacher recon | TS diff | overlay
    pixel_compare_{video}_{frame}.png            GT | teacher recon | student A TS | student B TS
    paper_3col_{video}_{frame}_{student}.png     GT | heatmap | overlay (paper style)
  curves/
    curve_{video}_{student}.png                  single student score vs GT
    curve_compare_{video}_{a}_vs_{b}.png         two students + similarity metrics
    curve_multi_{video}.png                      all students on one axes
  summary.json

Example:
  python util/visualize_vad_analysis.py \\
    --teacher_checkpoint output/avenue/author_teacher_mse_v2/checkpoint-best.pth \\
    --student_checkpoints \\
      mse=output/avenue/r0_mse_skip/checkpoint-best-student.pth \\
      bw2mse=output/avenue/author_teacher_bw2mse_a30_v2/checkpoint-best-student.pth \\
    --output_dir output/avenue/viz_vad_analysis \\
    --video_ids 01,06,12 \\
    --max_curve_videos 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util.abnormal_utils import filt


def parse_student_specs(specs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected name=path, got: {spec}")
        name, path = spec.split("=", 1)
        out[name.strip()] = path.strip()
    return out


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


def load_student_model(args, teacher_ckpt: str, student_ckpt: str, device: torch.device):
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


def fuse_ts_teacher_scores(ts_score, teacher_score, args) -> torch.Tensor:
    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    return w_teacher * teacher_score + w_ts * ts_score


def tensor_to_display_rgb(t: torch.Tensor) -> np.ndarray:
    """CHW tensor in [-1, 1] -> HWC [0, 1]."""
    x = t.detach().cpu().float().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    if x.dim() == 3:
        x = x.permute(1, 2, 0)
    return x.numpy()


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def channel_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel MSE over RGB channels. a,b: [3,H,W] -> [H,W]."""
    return ((a - b) ** 2).mean(dim=0)


@torch.no_grad()
def forward_pixel_maps(
    model,
    samples: torch.Tensor,
    grads: torch.Tensor,
    targets: torch.Tensor,
    mask_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Run Stage-2 forward and return unpatchified RGB error maps."""
    latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
    pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

    target_img = targets[:, :3]
    pred_teacher_img = model.unpatchify(pred_teacher)[:, :3]
    pred_stud_img = model.unpatchify(pred_stud)[:, :3]

    return {
        "target_rgb": target_img[0],
        "pred_teacher_rgb": pred_teacher_img[0],
        "pred_stud_rgb": pred_stud_img[0],
        "recon_teacher": channel_mse(target_img[0], pred_teacher_img[0]),
        "recon_student": channel_mse(target_img[0], pred_stud_img[0]),
        "ts_diff": channel_mse(pred_teacher_img[0], pred_stud_img[0]),
    }


@torch.no_grad()
def forward_frame_scores(
    model,
    samples: torch.Tensor,
    grads: torch.Tensor,
    targets: torch.Tensor,
    mask_ratio: float,
    args,
) -> Tuple[float, float, float]:
    """Return fused, ts, teacher scalar scores for one frame."""
    _, _, _, recon_error = model(
        samples, targets=targets, grad_mask=grads, mask_ratio=mask_ratio
    )
    if isinstance(recon_error, (list, tuple)):
        ts_score = recon_error[0]
        teacher_score = recon_error[1]
        fused = fuse_ts_teacher_scores(ts_score, teacher_score, args)
        return (
            float(fused[0].item()),
            float(ts_score[0].item()),
            float(teacher_score[0].item()),
        )
    return float(recon_error[0].item()), float(recon_error[0].item()), 0.0


def calculate_similarity_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return {"correlation": 0.0, "mae": float(np.mean(np.abs(a[:n] - b[:n]))) if n else 0.0}
    a, b = a[:n], b[:n]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
    mae = float(np.mean(np.abs(a - b)))
    return {"correlation": corr, "mae": mae}


def smooth_scores(scores: np.ndarray, args) -> np.ndarray:
    if args.curve_sigma and args.curve_sigma > 0:
        return gaussian_filter1d(scores, sigma=args.curve_sigma)
    if args.use_filt:
        return filt(
            scores,
            range=getattr(args, "smooth_range", 38 if args.dataset == "avenue" else 900),
            mu=getattr(args, "smooth_mu", 11 if args.dataset == "avenue" else 282),
        )
    return scores


def indices_for_video(dataset, video_id: str, label: Optional[int] = None) -> List[int]:
    """Use path/label metadata only — avoid __getitem__ (loads 4 images per index)."""
    out = []
    for i in range(len(dataset)):
        vid = dataset.data[i].split("/")[-2]
        if str(vid) != str(video_id):
            continue
        if label is None or int(dataset.labels[i]) == label:
            out.append(i)
    return out


def indices_for_videos(dataset, video_ids: List[str]) -> List[int]:
    """Metadata-only index lookup for one or more videos."""
    wanted = {str(v) for v in video_ids}
    return [
        i for i in range(len(dataset))
        if dataset.data[i].split("/")[-2] in wanted
    ]


def auto_pick_frame_indices(dataset, video_ids: List[str], n_abnormal: int, n_normal: int) -> Dict[str, List[int]]:
    picked: Dict[str, List[int]] = {}
    for vid in video_ids:
        ab = indices_for_video(dataset, vid, label=1)[:n_abnormal]
        nm = indices_for_video(dataset, vid, label=0)[:n_normal]
        picked[vid] = ab + nm
    return picked


def save_pixel_4panel(
    maps: Dict[str, torch.Tensor],
    out_path: Path,
    title: str,
    student_name: str,
):
    gt = tensor_to_display_rgb(maps["target_rgb"])
    recon = normalize_map(maps["recon_teacher"].cpu().numpy())
    ts = normalize_map(maps["ts_diff"].cpu().numpy())
    overlay = gt.copy()
    heat = cm.jet(ts)[..., :3]
    overlay = np.clip(0.6 * gt + 0.4 * heat, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gt)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(recon, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Teacher recon error")
    axes[2].imshow(ts, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"TS diff ({student_name})")
    axes[3].imshow(overlay)
    axes[3].set_title("TS overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_paper_3col(maps: Dict[str, torch.Tensor], out_path: Path):
    gt = tensor_to_display_rgb(maps["target_rgb"])
    ts = normalize_map(maps["ts_diff"].cpu().numpy())
    heat = cm.jet(ts)[..., :3]
    sep = np.zeros((gt.shape[0], 4, 3))
    paper = np.concatenate([gt, sep, heat], axis=1)
    plt.imsave(out_path, np.clip(paper, 0, 1))


def save_pixel_student_compare(
    gt_maps: Dict[str, torch.Tensor],
    student_maps: Dict[str, Dict[str, torch.Tensor]],
    out_path: Path,
    frame_name: str,
):
    gt = tensor_to_display_rgb(gt_maps["target_rgb"])
    recon = normalize_map(gt_maps["recon_teacher"].cpu().numpy())
    names = list(student_maps.keys())
    n = 2 + len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(gt)
    axes[0].set_title("GT")
    axes[1].imshow(recon, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Teacher recon")
    for ax, name in zip(axes[2:], names):
        ts = normalize_map(student_maps[name]["ts_diff"].cpu().numpy())
        ax.imshow(ts, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"TS diff ({name})")
        ax.axis("off")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.suptitle(frame_name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gt_contiguous_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    labels = np.asarray(labels).astype(bool)
    segments: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(labels):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def add_gt_shading(ax, labels: np.ndarray, frames: np.ndarray):
    """Highlight GT abnormal segments on the score axes."""
    segments = gt_contiguous_segments(labels)
    for i, (s, e) in enumerate(segments):
        ax.axvspan(
            frames[s],
            frames[e],
            color="#ff6666",
            alpha=0.22,
            zorder=0,
            label="GT abnormal" if i == 0 else None,
        )
    return segments


def plot_gt_strip(ax, labels: np.ndarray, frames: np.ndarray):
    """Binary GT strip below score curve."""
    labels = np.asarray(labels).astype(int)
    ax.fill_between(frames, 0, labels, step="mid", color="#e74c3c", alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["normal", "abnormal"], fontsize=8)
    ax.set_ylabel("GT", fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)


def plot_single_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    video_id: str,
    student_name: str,
    out_path: Path,
    args,
):
    scores = smooth_scores(scores, args)
    frames = np.arange(len(scores))
    fig, (ax, ax_gt) = plt.subplots(
        2, 1, figsize=(12, 4.2), sharex=True, gridspec_kw={"height_ratios": [3.5, 1]}
    )
    if labels is not None and len(labels) == len(scores):
        add_gt_shading(ax, labels, frames)
        plot_gt_strip(ax_gt, labels, frames)
        n_ab = int(np.sum(np.asarray(labels).astype(bool)))
        ax_gt.set_xlabel(f"Frame index  (GT abnormal: {n_ab}/{len(labels)} frames)")
    else:
        ax_gt.set_xlabel("Frame index")
    ax.plot(frames, scores, linewidth=2, label=f"{student_name} fused score", zorder=2)
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Video {video_id} — {student_name}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_curves(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    video_id: str,
    name_a: str,
    name_b: str,
    metrics: Dict[str, float],
    out_path: Path,
    args,
):
    n = min(len(scores_a), len(scores_b), len(labels) if labels is not None else len(scores_a))
    sa = smooth_scores(scores_a[:n], args)
    sb = smooth_scores(scores_b[:n], args)
    labels = labels[:n] if labels is not None else None
    frames = np.arange(n)

    fig, (ax, ax_gt) = plt.subplots(
        2, 1, figsize=(12, 4.5), sharex=True, gridspec_kw={"height_ratios": [3.5, 1]}
    )
    if labels is not None:
        add_gt_shading(ax, labels, frames)
        plot_gt_strip(ax_gt, labels, frames)
        n_ab = int(np.sum(np.asarray(labels).astype(bool)))
        ax_gt.set_xlabel(f"Frame index  (GT abnormal: {n_ab}/{n} frames)")
    else:
        ax_gt.set_xlabel("Frame index")
    ax.plot(frames, sa, "--", linewidth=2, color="tab:blue", alpha=0.85, label=name_a, zorder=2)
    ax.plot(frames, sb, "-", linewidth=2.5, color="tab:green", label=name_b, zorder=2)
    ax.set_ylabel("Fused anomaly score")
    ax.set_title(
        f"Video {video_id}: {name_a} vs {name_b}  "
        f"(corr={metrics['correlation']:.3f}, MAE={metrics['mae']:.4f})"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_student(
    all_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    video_id: str,
    out_path: Path,
    args,
):
    n = min(len(v) for v in all_scores.values())
    if labels is not None:
        n = min(n, len(labels))
    frames = np.arange(n)
    fig, (ax, ax_gt) = plt.subplots(
        2, 1, figsize=(12, 4.8), sharex=True, gridspec_kw={"height_ratios": [3.5, 1]}
    )
    if labels is not None:
        add_gt_shading(ax, labels[:n], frames)
        plot_gt_strip(ax_gt, labels[:n], frames)
        n_ab = int(np.sum(np.asarray(labels[:n]).astype(bool)))
        ax_gt.set_xlabel(f"Frame index  (GT abnormal: {n_ab}/{n} frames)")
    else:
        ax_gt.set_xlabel("Frame index")
    for name, scores in all_scores.items():
        s = smooth_scores(scores[:n], args)
        ax.plot(frames, s, linewidth=2, label=name, zorder=2)
    ax.set_ylabel("Fused anomaly score")
    ax.set_title(f"Video {video_id} — student comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_video_score_series(
    model,
    dataset,
    device,
    mask_ratio: float,
    args,
    max_frames: int = 0,
    video_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """video_id -> {fused, ts, teacher} lists aligned with frame order."""
    by_video: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"fused": [], "ts": [], "teacher": [], "labels": []}
    )
    if video_ids:
        frame_indices = indices_for_videos(dataset, video_ids)
    else:
        n_total = len(dataset) if max_frames <= 0 else min(len(dataset), max_frames)
        frame_indices = list(range(n_total))
    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    n = len(frame_indices)
    for j, i in enumerate(frame_indices):
        if j == 0 or (j + 1) % 100 == 0 or j + 1 == n:
            print(f"    {j + 1}/{n} frames", flush=True)
        samples, grads, targets, label, vid, _ = dataset[i]
        samples = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets = torch.from_numpy(targets).unsqueeze(0).to(device)
        fused, ts, teacher = forward_frame_scores(model, samples, grads, targets, mask_ratio, args)
        by_video[str(vid)]["fused"].append(fused)
        by_video[str(vid)]["ts"].append(ts)
        by_video[str(vid)]["teacher"].append(teacher)
        by_video[str(vid)]["labels"].append(int(label))
    return by_video


def main():
    parser = argparse.ArgumentParser(description="AED-MAE pixel heatmaps + score curves")
    parser.add_argument("--dataset", default="avenue", choices=["avenue", "shanghai"])
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument(
        "--student_checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video_ids", type=str, default="01,06,12,14,21")
    parser.add_argument("--frame_indices", type=str, default=None, help="global dataset indices, comma-sep")
    parser.add_argument("--n_abnormal_frames", type=int, default=2, help="auto-pick abnormal frames per video")
    parser.add_argument("--n_normal_frames", type=int, default=1, help="auto-pick normal frames per video")
    parser.add_argument("--max_curve_videos", type=int, default=0, help="0 = all listed video_ids")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all test frames for score collection")
    parser.add_argument("--curve_sigma", type=float, default=0.0, help="gaussian smooth; 0 disables")
    parser.add_argument("--use_filt", action="store_true", help="use Avenue/Shanghai filt() smoothing on curves")
    parser.add_argument("--skip_pixel", action="store_true")
    parser.add_argument("--skip_curves", action="store_true")
    args_cli = parser.parse_args()

    args = get_configs_avenue() if args_cli.dataset == "avenue" else get_configs_shanghai()
    args.curve_sigma = args_cli.curve_sigma
    args.use_filt = args_cli.use_filt
    mask_ratio = args_cli.mask_ratio if args_cli.mask_ratio is not None else args.mask_ratio
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args_cli.output_dir)
    pixel_dir = out_dir / "pixel"
    curve_dir = out_dir / "curves"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    dataset = AbnormalDatasetGradientsTest(args)
    student_specs = parse_student_specs(args_cli.student_checkpoints)
    video_ids = [v.strip() for v in args_cli.video_ids.split(",") if v.strip()]
    if args_cli.max_curve_videos > 0:
        video_ids = video_ids[: args_cli.max_curve_videos]

    models: Dict[str, torch.nn.Module] = {}
    score_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    print("Loading models ...")
    for name, ckpt in student_specs.items():
        print(f"  student={name}  ckpt={ckpt}")
        models[name] = load_student_model(args, args_cli.teacher_checkpoint, ckpt, device)

    if not args_cli.skip_curves:
        n_score_frames = len(indices_for_videos(dataset, video_ids))
        print(
            f"Collecting per-frame scores for {len(video_ids)} video(s), "
            f"{n_score_frames} frame(s) x {len(student_specs)} student(s) ...",
            flush=True,
        )
        if args_cli.max_frames > 0 and args_cli.max_frames < n_score_frames:
            print(
                f"  [warn] --max_frames {args_cli.max_frames} < {n_score_frames}; "
                "GT abnormal region may be truncated (e.g. Avenue video 05 abnormal starts ~468)",
                flush=True,
            )
        for name in student_specs:
            print(f"  student={name}", flush=True)
            score_series[name] = collect_video_score_series(
                models[name],
                dataset,
                device,
                mask_ratio,
                args,
                args_cli.max_frames,
                video_ids=video_ids,
            )

    summary = {"similarity": {}, "videos": video_ids, "students": list(student_specs.keys())}

    if not args_cli.skip_curves:
        print("Plotting score curves ...")
        curve_dir.mkdir(parents=True, exist_ok=True)
        for vid in video_ids:
            if vid not in score_series[list(student_specs.keys())[0]]:
                print(f"  [warn] video {vid} not in dataset, skip curves")
                continue
            labels = np.array(score_series[list(student_specs.keys())[0]][vid]["labels"])
            multi = {n: np.array(score_series[n][vid]["fused"]) for n in student_specs}
            plot_multi_student(
                multi, labels, vid, curve_dir / f"curve_multi_{vid}.png", args
            )
            for name in student_specs:
                plot_single_curve(
                    np.array(score_series[name][vid]["fused"]),
                    labels,
                    vid,
                    name,
                    curve_dir / f"curve_{vid}_{name}.png",
                    args,
                )
            names = list(student_specs.keys())
            if len(names) >= 2:
                a, b = names[0], names[1]
                sa = np.array(score_series[a][vid]["fused"])
                sb = np.array(score_series[b][vid]["fused"])
                metrics = calculate_similarity_metrics(sa, sb)
                summary["similarity"][vid] = {f"{a}_vs_{b}": metrics}
                plot_compare_curves(
                    sa, sb, labels, vid, a, b, metrics,
                    curve_dir / f"curve_compare_{vid}_{a}_vs_{b}.png",
                    args,
                )

    if not args_cli.skip_pixel:
        print("Generating pixel heatmaps ...")
        if args_cli.frame_indices:
            indices = [int(x) for x in args_cli.frame_indices.split(",")]
            frame_pick = {"_manual": indices}
        else:
            frame_pick = auto_pick_frame_indices(
                dataset, video_ids, args_cli.n_abnormal_frames, args_cli.n_normal_frames
            )
        n_frames = sum(len(v) for v in frame_pick.values())
        print(f"  picked {n_frames} frame(s) across {len(frame_pick)} video(s)")

        done = 0
        for vid, indices in frame_pick.items():
            for idx in indices:
                done += 1
                samples, grads, targets, label, video, frame_path = dataset[idx]
                frame_name = Path(frame_path).name
                print(f"  [{done}/{n_frames}] video={video} frame={frame_name} label={label}")
                samples_t = torch.from_numpy(samples).unsqueeze(0).to(device)
                grads_t = torch.from_numpy(grads).unsqueeze(0).to(device)
                targets_t = torch.from_numpy(targets).unsqueeze(0).to(device)

                student_maps: Dict[str, Dict[str, torch.Tensor]] = {}
                gt_maps = None
                for name, model in models.items():
                    maps = forward_pixel_maps(
                        model, samples_t, grads_t, targets_t, mask_ratio
                    )
                    student_maps[name] = maps
                    if gt_maps is None:
                        gt_maps = maps
                    save_pixel_4panel(
                        maps,
                        pixel_dir / f"pixel_4panel_{video}_{frame_name}_{name}.png",
                        f"Video {video} frame {frame_name} label={label} ({name})",
                        name,
                    )
                    save_paper_3col(
                        maps,
                        pixel_dir / f"paper_3col_{video}_{frame_name}_{name}.png",
                    )

                if gt_maps and len(student_maps) >= 2:
                    save_pixel_student_compare(
                        gt_maps,
                        student_maps,
                        pixel_dir / f"pixel_compare_{video}_{frame_name}.png",
                        f"Video {video} {frame_name} (label={label})",
                    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
