"""
Parse all Avenue experiment logs and produce comparison tables + learning curves.

Usage:
  python util/analyze_experiments.py --output_dir output/avenue/exp_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Human-readable metadata inferred from experiment folder names.
EXP_META: Dict[str, Dict[str, str]] = {
    "paper_baseline": {
        "loss": "mse",
        "train": "full",
        "notes": "CVPR paper: Adam, cls+Eq.6, 100+40ep",
    },
    "e0_all": {"loss": "mse", "train": "full", "notes": "baseline MSE, distill all"},
    "e1_skip": {"loss": "mse", "train": "full", "notes": "MSE, skip abnormal in stage2"},
    "e2_margin": {"loss": "margin", "train": "full", "notes": "margin loss (failed)"},
    "bw2_all": {"loss": "bw2", "train": "full", "notes": "pure BW2 (collapsed)"},
    "bw2_all_v2": {"loss": "bw2", "train": "full", "notes": "pure BW2 retry"},
    "bw2mse_a30_all": {"loss": "bw2_mse", "train": "full", "notes": "BW2+MSE alpha=0.3"},
    "author_teacher_mse": {"loss": "mse", "train": "student_only", "notes": "bad teacher ckpt"},
    "author_teacher_mse_v2": {"loss": "mse", "train": "student_only", "notes": "good teacher"},
    "author_teacher_bw2mse_a30_v2": {"loss": "bw2_mse", "train": "student_only", "notes": "BW2+MSE alpha=0.3"},
    "r0_mse_skip": {"loss": "mse", "train": "student_only", "notes": "repro baseline"},
    "r1_bw2mse_norm_skip": {"loss": "bw2_mse_norm", "train": "student_only", "notes": "normalized BW2+MSE"},
    "r2_bw2lr_mse_skip": {"loss": "bw2_lowrank_mse", "train": "student_only", "notes": "low-rank BW2+MSE"},
    "t1_clip8_mse_skip": {"loss": "mse", "train": "student_only", "notes": "clip_len=8, MSE only"},
    "t3_joint_bw2mse_k8": {"loss": "temporal_joint_bw2mse", "train": "student_only", "notes": "no cls/fusion"},
    "t3_joint_bw2mse_k8_cls": {"loss": "temporal_joint_bw2mse", "train": "student_only", "notes": "with cls+fusion"},
    "cls_bw2mse_a30": {"loss": "bw2_mse", "train": "student_only", "notes": "cls+fusion"},
    "cls_mse_tw": {"loss": "mse_tw", "train": "student_only", "notes": "teacher-weighted MSE"},
    "cls_contrastive": {"loss": "contrastive", "train": "student_only", "notes": "gap contrastive"},
    "spa_a_map": {"loss": "mse", "train": "full", "notes": "Route-A: map only"},
    "spa_a_fg": {"loss": "mse", "train": "student_only", "notes": "Route-A: fg only"},
    "spa_a_attn": {"loss": "mse", "train": "student_only", "notes": "Route-A: attn v1 (broken infer)"},
    "spa_a_attn_v2": {"loss": "mse", "train": "student_only", "notes": "Route-A: attn v2 fixed"},
    "spa_a_all": {"loss": "mse", "train": "full", "notes": "Route-A: map+fg+attn"},
    "spa_b_peak9": {"loss": "mse", "train": "infer_only", "notes": "Route-B: temporal peak w=9"},
    "spa_b_topk8": {"loss": "mse", "train": "student_only", "notes": "Route-B: top-k patch k=8"},
    "spa_b_hn": {"loss": "mse", "train": "student_only", "notes": "Route-B: hard normal mining"},
    "spa_b_topk8_hn": {"loss": "mse", "train": "student_only", "notes": "Route-B: topk+hn"},
    "spa_b_topk8_peak9": {"loss": "mse", "train": "student_only", "notes": "Route-B: topk+peak eval"},
}


def parse_log(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def best_metrics(rows: List[Dict], stage2_only: bool = False) -> Tuple[float, float, int]:
    filtered = [r for r in rows if not stage2_only or r.get("epoch", 0) >= 100]
    if not filtered:
        return 0.0, 0.0, -1
    best = max(filtered, key=lambda r: r.get("test_micro", 0))
    return (
        float(best.get("test_micro", 0)),
        float(best.get("test_macro", 0)),
        int(best.get("epoch", -1)),
    )


def stage2_gain(rows: List[Dict]) -> float:
    pre = [r for r in rows if r.get("epoch", 0) < 100]
    post = [r for r in rows if r.get("epoch", 0) >= 100]
    if not pre or not post:
        return 0.0
    best_pre = max(pre, key=lambda r: r.get("test_micro", 0))
    best_post = max(post, key=lambda r: r.get("test_micro", 0))
    return float(best_post["test_micro"] - best_pre["test_micro"])


def collect_experiments(base: Path) -> List[Dict]:
    results = []
    for log_path in sorted(base.glob("*/log_test.txt")):
        name = log_path.parent.name
        rows = parse_log(log_path)
        if not rows:
            continue
        micro_all, macro_all, ep_all = best_metrics(rows, stage2_only=False)
        micro_s2, macro_s2, ep_s2 = best_metrics(rows, stage2_only=True)
        meta = EXP_META.get(name, {"loss": "?", "train": "?", "notes": ""})
        results.append(
            {
                "name": name,
                "loss": meta["loss"],
                "train": meta["train"],
                "notes": meta["notes"],
                "best_micro_all": micro_all,
                "best_macro_all": macro_all,
                "best_epoch_all": ep_all,
                "best_micro_s2": micro_s2,
                "best_macro_s2": macro_s2,
                "best_epoch_s2": ep_s2,
                "stage2_gain": stage2_gain(rows),
                "n_epochs": max(r.get("epoch", 0) for r in rows),
                "rows": rows,
            }
        )
    return results


def plot_bar_comparison(exps: List[Dict], out_path: Path):
    exps = sorted(exps, key=lambda e: e["best_micro_s2"], reverse=True)
    names = [e["name"] for e in exps]
    micro = [e["best_micro_s2"] * 100 for e in exps]
    macro = [e["best_macro_s2"] * 100 for e in exps]
    colors = []
    for e in exps:
        if e["loss"] in ("bw2",):
            colors.append("#e74c3c")
        elif e["loss"] == "contrastive":
            colors.append("#f39c12")
        elif "temporal" in e["loss"]:
            colors.append("#9b59b6")
        elif e["loss"] == "mse":
            colors.append("#3498db")
        else:
            colors.append("#2ecc71")

    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(names))))
    y = np.arange(len(names))
    ax.barh(y - 0.15, micro, height=0.3, color=colors, alpha=0.85, label="micro-AUC (stage2 best)")
    ax.barh(y + 0.15, macro, height=0.3, color=colors, alpha=0.45, label="macro-AUC (stage2 best)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUC (%)")
    ax.set_title("Avenue experiments — best Stage-2 performance")
    ax.axvline(90.0, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(35, 92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(exps: List[Dict], out_path: Path, highlight: Optional[List[str]] = None):
    highlight = highlight or [
        "r0_mse_skip",
        "author_teacher_bw2mse_a30_v2",
        "cls_contrastive",
        "bw2_all",
        "t3_joint_bw2mse_k8_cls",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for e in exps:
        if e["name"] not in highlight:
            continue
        epochs = [r["epoch"] for r in e["rows"]]
        micro = [r["test_micro"] * 100 for r in e["rows"]]
        macro = [r["test_macro"] * 100 for r in e["rows"]]
        axes[0].plot(epochs, micro, linewidth=1.5, label=e["name"])
        axes[1].plot(epochs, macro, linewidth=1.5, label=e["name"])
    for ax in axes:
        ax.axvline(100, color="gray", linestyle="--", alpha=0.6, label="Stage 2 start")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("micro-AUC (%)")
    axes[0].set_title("Micro-AUC learning curves")
    axes[1].set_ylabel("macro-AUC (%)")
    axes[1].set_title("Macro-AUC learning curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stage2_gain(exps: List[Dict], out_path: Path):
    valid = [e for e in exps if e["stage2_gain"] != 0 and e["best_micro_s2"] > 0.5]
    valid = sorted(valid, key=lambda e: e["stage2_gain"], reverse=True)
    names = [e["name"] for e in valid]
    gains = [e["stage2_gain"] * 100 for e in valid]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(names))))
    colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gains]
    ax.barh(names, gains, color=colors, alpha=0.85)
    ax.set_xlabel("micro-AUC gain (Stage2 best − Stage1 best) [%]")
    ax.set_title("Stage-2 distillation gain")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_family(exps: List[Dict], out_path: Path):
    """Group by loss family, show distribution of stage2 micro-AUC."""
    families: Dict[str, List[float]] = {}
    for e in exps:
        if e["best_micro_s2"] < 0.5:
            continue
        fam = e["loss"]
        families.setdefault(fam, []).append(e["best_micro_s2"] * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(families.keys(), key=lambda k: np.mean(families[k]), reverse=True)
    data = [families[k] for k in labels]
    bp = ax.boxplot(data, labels=labels, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax.set_ylabel("best stage2 micro-AUC (%)")
    ax.set_title("Loss family comparison")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_table(exps: List[Dict], out_path: Path):
    exps = sorted(exps, key=lambda e: e["best_micro_s2"], reverse=True)
    lines = [
        "# Avenue Experiment Summary\n",
        "| Run | Loss | Train | Best micro (S2) | Best macro (S2) | S2 gain | Notes |",
        "|-----|------|-------|-----------------|-----------------|---------|-------|",
    ]
    for e in exps:
        lines.append(
            f"| {e['name']} | {e['loss']} | {e['train']} "
            f"| {e['best_micro_s2']*100:.2f}% | {e['best_macro_s2']*100:.2f}% "
            f"| {e['stage2_gain']*100:+.2f}% | {e['notes']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(ROOT / "output" / "avenue"))
    parser.add_argument("--output_dir", default=str(ROOT / "output" / "avenue" / "exp_analysis"))
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exps = collect_experiments(base)
    summary = [{k: v for k, v in e.items() if k != "rows"} for e in exps]
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_markdown_table(exps, out / "summary.md")
    plot_bar_comparison(exps, out / "fig_bar_comparison.png")
    plot_learning_curves(exps, out / "fig_learning_curves.png")
    plot_stage2_gain(exps, out / "fig_stage2_gain.png")
    plot_loss_family(exps, out / "fig_loss_family.png")

    print(f"Analyzed {len(exps)} experiments -> {out}")
    top3 = sorted(exps, key=lambda e: e["best_micro_s2"], reverse=True)[:3]
    for e in top3:
        print(f"  {e['name']}: micro={e['best_micro_s2']*100:.2f}% macro={e['best_macro_s2']*100:.2f}%")


if __name__ == "__main__":
    main()
