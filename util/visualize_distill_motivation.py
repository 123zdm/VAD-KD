"""
Motivation / analysis figures for VAD knowledge distillation.

Compares MSE vs BW2 student checkpoints on Avenue test set and produces:
  1) Normal vs abnormal score separation (KDE)
  2) Teacher-student mu/std alignment (CDF)
  3) Patch correlation structure (heatmap)
  4) Masked patch t-SNE (teacher vs students)
  5) Temporal patch-stat correlation within a normal clip

Example:
  python util/visualize_distill_motivation.py \\
    --teacher_checkpoint output/avenue/author_teacher_mse_v2/checkpoint-best.pth \\
    --student_checkpoints \\
      mse=output/avenue/author_teacher_mse_v2/checkpoint-best-student.pth \\
      bw2=output/avenue/author_teacher_bw2mse_a30_v2/checkpoint-best-student.pth \\
    --output_dir output/avenue/viz_motivation_v2 \\
    --max_frames 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8


@dataclass
class FrameStats:
    label: int
    video: str
    st_mse: float
    bw2: float
    mu_gap: float
    std_gap: float
    corr_gap: float
    anomaly_score: float


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


def diagonal_bw2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> float:
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    std_x = x.std(dim=0, unbiased=False).clamp(min=eps)
    std_y = y.std(dim=0, unbiased=False).clamp(min=eps)
    mean_term = ((mu_x - mu_y) ** 2).mean()
    cov_term = ((std_x - std_y) ** 2).mean()
    return float((mean_term + cov_term).item())


def corr_matrix(feats: torch.Tensor, max_dims: int = 64) -> np.ndarray:
    x = feats
    if x.shape[1] > max_dims:
        idx = torch.linspace(0, x.shape[1] - 1, max_dims).long()
        x = x[:, idx]
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
    c = (x.T @ x) / max(x.shape[0] - 1, 1)
    return c.detach().cpu().numpy()


def extract_frame_stats(
    model,
    samples: torch.Tensor,
    grads: torch.Tensor,
    targets: torch.Tensor,
    label: int,
    video: str,
    mask_ratio: float,
    corr_dims: int,
) -> FrameStats:
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
        pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

        idx = mask[0].bool()
        z_s = pred_stud[0, idx]
        z_t = pred_teacher[0, idx]

        st_mse = float((((z_s - z_t) ** 2).mean()).item())
        bw2 = diagonal_bw2(z_s, z_t, eps=model.bw2_eps)
        mu_gap = float((z_s.mean(0) - z_t.mean(0)).pow(2).mean().sqrt().item())
        std_gap = float((z_s.std(0, unbiased=False) - z_t.std(0, unbiased=False)).pow(2).mean().sqrt().item())

        c_s = corr_matrix(z_s, max_dims=corr_dims)
        c_t = corr_matrix(z_t, max_dims=corr_dims)
        corr_gap = float(np.linalg.norm(c_s - c_t, ord="fro") / c_t.size)

        _, _, _, score = model(
            samples, targets=targets, grad_mask=grads, mask_ratio=mask_ratio
        )
        if isinstance(score, (list, tuple)):
            score = score[0]
        anomaly_score = float(score[0].item())

    return FrameStats(
        label=int(label),
        video=str(video),
        st_mse=st_mse,
        bw2=bw2,
        mu_gap=mu_gap,
        std_gap=std_gap,
        corr_gap=corr_gap,
        anomaly_score=anomaly_score,
    )


def collect_stats(
    model,
    dataset,
    device,
    mask_ratio: float,
    max_frames: int,
    seed: int,
    corr_dims: int,
) -> List[FrameStats]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if max_frames < len(indices):
        indices = rng.choice(indices, size=max_frames, replace=False)

    rows: List[FrameStats] = []
    for i, index in enumerate(indices):
        samples, grads, targets, label, video, _ = dataset[int(index)]
        samples = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads = torch.from_numpy(grads).unsqueeze(0).to(device)
        targets = torch.from_numpy(targets).unsqueeze(0).to(device)
        rows.append(
            extract_frame_stats(
                model, samples, grads, targets, label, video, mask_ratio, corr_dims
            )
        )
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(indices)} frames")
    return rows


def plot_score_kde(all_stats: Dict[str, List[FrameStats]], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(
        axes,
        ["anomaly_score", "st_mse"],
        ["Fused / ST anomaly score", "Masked patch MSE(S, T)"],
    ):
        for name, rows in all_stats.items():
            normal = [getattr(r, metric) for r in rows if r.label == 0]
            abnormal = [getattr(r, metric) for r in rows if r.label == 1]
            if normal:
                ax.hist(
                    normal,
                    bins=40,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    label=f"{name} normal",
                )
            if abnormal:
                ax.hist(
                    abnormal,
                    bins=40,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    linestyle="--",
                    label=f"{name} abnormal",
                )
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_alignment_cdf(all_stats: Dict[str, List[FrameStats]], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [("mu_gap", "||mu_S - mu_T||"), ("std_gap", "||std_S - std_T||"), ("bw2", "BW2(S,T)")]
    for ax, (metric, title) in zip(axes, metrics):
        for name, rows in all_stats.items():
            vals = sorted(getattr(r, metric) for r in rows if r.label == 0)
            if not vals:
                continue
            y = np.linspace(0, 1, len(vals), endpoint=False)
            ax.plot(vals, y, label=name, linewidth=1.8)
        ax.set_title(f"Normal frames: {title}")
        ax.set_xlabel("value")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pick_normal_video_name(dataset) -> str:
    for i in range(len(dataset)):
        _, _, _, label, video, _ = dataset[i]
        if int(label) == 0:
            return str(video)
    return "01"


def plot_correlation_heatmap(
    model,
    dataset,
    device,
    mask_ratio: float,
    corr_dims: int,
    out_path: Path,
    video_name: str = "01",
):
    target_idx = None
    for i in range(len(dataset)):
        _, _, _, label, video, _ = dataset[i]
        if label == 0 and str(video) == video_name:
            target_idx = i
            break
    if target_idx is None:
        print(f"[warn] video {video_name} not found; skip correlation heatmap")
        return

    samples, grads, targets, _, _, _ = dataset[target_idx]
    samples = torch.from_numpy(samples).unsqueeze(0).to(device)
    grads = torch.from_numpy(grads).unsqueeze(0).to(device)
    targets = torch.from_numpy(targets).unsqueeze(0).to(device)

    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
        pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)
        idx = mask[0].bool()
        z_s = pred_stud[0, idx]
        z_t = pred_teacher[0, idx]

    c_t = corr_matrix(z_t, max_dims=corr_dims)
    c_s = corr_matrix(z_s, max_dims=corr_dims)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    mats = [c_t, c_s, np.abs(c_t - c_s)]
    titles = ["Teacher corr", "Student corr", "|Teacher - Student|"]
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Masked patch correlation structure (video {video_name}, normal frame)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tsne(all_models: Dict[str, torch.nn.Module], dataset, device, mask_ratio, out_path, max_patches=1200, seed=0):
    rng = np.random.default_rng(seed)
    feats = []
    tags = []
    for name, model in all_models.items():
        count = 0
        for i in range(len(dataset)):
            _, _, _, label, _, _ = dataset[i]
            if label != 0:
                continue
            samples, grads, targets, _, _, _ = dataset[i]
            samples = torch.from_numpy(samples).unsqueeze(0).to(device)
            grads = torch.from_numpy(grads).unsqueeze(0).to(device)
            with torch.no_grad():
                latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
                if name == "teacher":
                    pred = model.forward_decoder(latent, ids_restore)
                else:
                    pred, _ = model.forward_decoder_TS(latent, ids_restore)
                idx = mask[0].bool()
                z = pred[0, idx].detach().cpu().numpy()
            take = min(z.shape[0], max(1, max_patches // max(len(all_models), 1)))
            pick = rng.choice(z.shape[0], size=take, replace=False)
            feats.append(z[pick])
            tags.extend([name] * take)
            count += take
            if count >= max_patches // max(len(all_models), 1):
                break

    if not feats:
        print("[warn] no normal patches for t-SNE")
        return

    x = np.concatenate(feats, axis=0)
    if x.shape[0] < 20:
        print("[warn] too few patches for t-SNE")
        return
    emb = TSNE(n_components=2, perplexity=min(30, x.shape[0] - 1), random_state=seed, init="pca").fit_transform(x)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in all_models:
        m = np.array(tags) == name
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, label=name)
    ax.legend()
    ax.set_title("t-SNE of masked patch features (normal frames)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temporal_covariance(model, dataset, device, mask_ratio, out_path, video_name="01", max_frames=40):
    seq_stats = []
    for i in range(len(dataset)):
        samples, grads, targets, label, video, _ = dataset[i]
        if label != 0 or str(video) != video_name:
            continue
        samples = torch.from_numpy(samples).unsqueeze(0).to(device)
        grads = torch.from_numpy(grads).unsqueeze(0).to(device)
        with torch.no_grad():
            latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
            pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)
            idx = mask[0].bool()
            z_s = pred_stud[0, idx]
            z_t = pred_teacher[0, idx]
        seq_stats.append(
            {
                "mu_t": z_t.mean(0).detach().cpu().numpy(),
                "mu_s": z_s.mean(0).detach().cpu().numpy(),
                "std_t": z_t.std(0, unbiased=False).detach().cpu().numpy(),
                "std_s": z_s.std(0, unbiased=False).detach().cpu().numpy(),
            }
        )
        if len(seq_stats) >= max_frames:
            break

    if len(seq_stats) < 5:
        print("[warn] not enough frames for temporal covariance plot")
        return

    def temporal_sim(key: str) -> np.ndarray:
        mat = np.stack([s[key] for s in seq_stats], axis=0)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        if mat.shape[0] < 2:
            return np.eye(max(mat.shape[0], 1))
        return cosine_similarity(mat)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    panels = [
        (temporal_sim("mu_t"), "Teacher temporal sim (patch mean)"),
        (temporal_sim("mu_s"), "Student temporal sim (patch mean)"),
        (temporal_sim("std_t"), "Teacher temporal sim (patch std)"),
        (temporal_sim("std_s"), "Student temporal sim (patch std)"),
    ]
    for ax, (mat, title) in zip(axes.flatten(), panels):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame index")
        ax.set_ylabel("frame index")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Temporal statistics correlation within normal clip {video_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_auc(all_stats: Dict[str, List[FrameStats]]) -> Dict[str, float]:
    out = {}
    for name, rows in all_stats.items():
        y = np.array([r.label for r in rows])
        s = np.array([r.anomaly_score for r in rows])
        if len(np.unique(y)) < 2:
            continue
        out[name] = float(roc_auc_score(y, s))
    return out


def main():
    parser = argparse.ArgumentParser(description="VAD distillation motivation visualizations")
    parser.add_argument("--dataset", default="avenue", choices=["avenue", "shanghai"])
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument(
        "--student_checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. mse=.../checkpoint-best-student.pth bw2=.../checkpoint-best-student.pth",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=800)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr_dims", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = get_configs_avenue() if args_cli.dataset == "avenue" else get_configs_shanghai()
    mask_ratio = args_cli.mask_ratio if args_cli.mask_ratio is not None else args.mask_ratio
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AbnormalDatasetGradientsTest(args)
    student_specs = parse_student_specs(args_cli.student_checkpoints)

    all_stats: Dict[str, List[FrameStats]] = {}
    loaded_models: Dict[str, torch.nn.Module] = {}

    print("Collecting per-frame statistics ...")
    for name, ckpt in student_specs.items():
        print(f"  student={name} ckpt={ckpt}")
        model = load_student_model(args, args_cli.teacher_checkpoint, ckpt, device)
        loaded_models[name] = model
        all_stats[name] = collect_stats(
            model, dataset, device, mask_ratio, args_cli.max_frames, args_cli.seed, args_cli.corr_dims
        )

    with open(out_dir / "frame_stats.json", "w", encoding="utf-8") as f:
        json.dump({k: [asdict(r) for r in v] for k, v in all_stats.items()}, f, indent=2)

    aucs = summarize_auc(all_stats)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"frame_auc": aucs, "num_frames": args_cli.max_frames}, f, indent=2)

    plot_score_kde(all_stats, out_dir / "fig1_score_separation.png")
    plot_alignment_cdf(all_stats, out_dir / "fig2_alignment_cdf.png")

    first_name = next(iter(loaded_models))
    video_name = pick_normal_video_name(dataset)
    plot_correlation_heatmap(
        loaded_models[first_name], dataset, device, mask_ratio, args_cli.corr_dims,
        out_dir / f"fig3_correlation_{first_name}.png",
        video_name=video_name,
    )
    for name, model in loaded_models.items():
        if name == first_name:
            continue
        plot_correlation_heatmap(
            model, dataset, device, mask_ratio, args_cli.corr_dims,
            out_dir / f"fig3_correlation_{name}.png",
            video_name=video_name,
        )

    teacher_only = build_model(args).to(device)
    teacher_only.load_state_dict(
        torch.load(args_cli.teacher_checkpoint, map_location="cpu", weights_only=False)["model"],
        strict=False,
    )
    teacher_only.eval()
    tsne_models = {"teacher": teacher_only}
    for name, model in loaded_models.items():
        tsne_models[name] = model
    plot_tsne(tsne_models, dataset, device, mask_ratio, out_dir / "fig4_tsne_patches.png", seed=args_cli.seed)

    for name, model in loaded_models.items():
        plot_temporal_covariance(
            model, dataset, device, mask_ratio,
            out_dir / f"fig5_temporal_cov_{name}.png",
            video_name=video_name,
        )

    print(f"Saved figures to {out_dir}")
    print(f"Frame-level AUC (unsmoothed): {aucs}")


if __name__ == "__main__":
    main()
