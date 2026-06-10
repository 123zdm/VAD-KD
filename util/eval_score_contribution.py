"""
Evaluate author checkpoint and decompose anomaly-score contributions.

Uses the same post-processing as inference.py / engine_train.py (filt + ROC).

Example:
  python util/eval_score_contribution.py \\
    --student_checkpoint ../aed-mae_search/output/avenue_cls_head/checkpoint-best-student.pth \\
    --teacher_checkpoint ../aed-mae_search/output/avenue_cls_head/checkpoint-best.pth \\
    --output_dir output/avenue/author_cls_head_contrib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import get_configs_avenue
from data.test_dataset import AbnormalDatasetGradientsTest
from inference import evaluate_model, fuse_ts_teacher_scores
from model.model_factory import mae_cvt_patch16
from util.visualize_paper_fig4 import remap_checkpoint_keys


def load_merged_model(args, teacher_ckpt: str, student_ckpt: str, device: torch.device):
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
def extract_scores(model, samples, grads, targets, mask_ratio, args) -> Dict[str, float]:
    latent, mask, ids_restore = model.forward_encoder(samples, mask_ratio, grads)
    pred_stud, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

    ts_scores, teacher_scores = model.abnormal_score_TS(
        targets, pred_stud, pred_teacher, mask, grads
    )
    ts_gap = float(ts_scores[0].item())
    teacher_recon = float(teacher_scores[0].item())
    official = float(
        fuse_ts_teacher_scores(
            torch.tensor([ts_gap]), torch.tensor([teacher_recon]), args
        )[0].item()
    )

    return {
        "ts_gap": ts_gap,
        "teacher_recon": teacher_recon,
        "official_fused": official,
    }


def per_video_auc(predictions, labels, videos) -> Dict[str, float]:
    out = {}
    for vid in np.unique(videos):
        m = videos == vid
        pred = predictions[m]
        lbl = labels[m]
        lbl_pad = np.array([0] + list(lbl) + [1])
        pred_pad = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl_pad, pred_pad)
        out[str(vid)] = float(metrics.auc(fpr, tpr))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = get_configs_avenue()
    mask_ratio = args_cli.mask_ratio if args_cli.mask_ratio is not None else args.mask_ratio
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AbnormalDatasetGradientsTest(args)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, pin_memory=False, shuffle=False
    )
    model = load_merged_model(
        args, args_cli.teacher_checkpoint, args_cli.student_checkpoint, device
    )

    streams: Dict[str, List[float]] = {
        k: [] for k in ["ts_gap", "teacher_recon", "official_fused"]
    }
    labels, videos = [], []

    print("Collecting scores on full Avenue test set ...")
    for i, (samples, grads, targets, label, vid, _) in enumerate(loader):
        if i % 500 == 0:
            print(f"  {i}/{len(loader)}")
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        sc = extract_scores(model, samples, grads, targets, mask_ratio, args)
        for k, v in sc.items():
            streams[k].append(v)
        labels.append(float(label.item()))
        videos.append(str(vid[0]))

    labels_arr = np.array(labels)
    videos_arr = np.array(videos)

    results = {"checkpoint": args_cli.student_checkpoint, "components": {}, "per_video": {}}
    print("\n=== AUC with official filt() post-processing ===")
    for name, preds in streams.items():
        preds_arr = np.array(preds)
        micro, macro = evaluate_model(
            preds_arr, labels_arr, videos_arr,
            range=args.smooth_range, mu=args.smooth_mu,
        )
        results["components"][name] = {
            "micro_auc": float(micro),
            "macro_auc": float(macro),
        }
        print(f"  {name:20s}  micro={micro*100:.2f}%  macro={macro*100:.2f}%")

    for score_name in ["official_fused", "ts_gap", "teacher_recon"]:
        preds = np.array(streams[score_name])
        pv = {}
        for vid in np.unique(videos_arr):
            m = videos_arr == vid
            pred = preds[m]
            pred = __import__("util.abnormal_utils", fromlist=["filt"]).filt(
                pred, range=args.smooth_range, mu=args.smooth_mu
            )
            lbl = labels_arr[m]
            lbl_pad = np.array([0] + list(lbl) + [1])
            pred_pad = np.array([0] + list(pred) + [1])
            fpr, tpr, _ = metrics.roc_curve(lbl_pad, pred_pad)
            pv[str(vid)] = float(metrics.auc(fpr, tpr))
        results["per_video"][score_name] = pv

    with open(out_dir / "contribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # markdown report
    lines = [
        "# Author Model Score Contribution (avenue_cls_head)\n",
        f"Checkpoint: `{args_cli.student_checkpoint}`\n",
        "## Full-test AUC (with filt)\n",
        "| Component | Micro AUC | Macro AUC |",
        "|-----------|-----------|-----------|",
    ]
    for name, v in results["components"].items():
        lines.append(
            f"| {name} | {v['micro_auc']*100:.2f}% | {v['macro_auc']*100:.2f}% |"
        )
    (out_dir / "contribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
