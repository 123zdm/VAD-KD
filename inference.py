from collections.abc import Iterable

import numpy as np
import torch
from sklearn import metrics

from util import misc
from util.abnormal_utils import filt
from util.score_postprocess import apply_temporal_peak_pooling


def fuse_ts_teacher_scores(ts_score, teacher_score, args):
    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    return w_teacher * teacher_score + w_ts * ts_score


def _configure_infer_model(model, args):
    student_only = bool(getattr(args, "student_infer_only", False))
    model.student_infer_only = student_only
    if student_only:
        model.train_TS = False
        print("Inference mode: student-only (score = student reconstruction error)")
        return
    model.train_TS = True
    if args.dataset == 'avenue':
        model.abnormal_score_func_TS = "L2"
    else:
        model.abnormal_score_func_TS = 'L1'
    print(
        "Inference mode: teacher+student "
        f"({getattr(args, 'score_weight_teacher', 0.4)}·teacher + "
        f"{getattr(args, 'score_weight_ts', 0.3)}·ts_gap)"
    )


def inference(model: torch.nn.Module, data_loader: Iterable,
              device: torch.device,
              log_writer=None, args=None):
    model.eval()
    _configure_infer_model(model, args)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions = []
    labels = []
    videos = []
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        _, _, _, recon_error = model(
            samples, targets=targets, grad_mask=grads, mask_ratio=args.mask_ratio
        )
        if getattr(args, "student_infer_only", False):
            frame_scores = recon_error.detach().cpu().numpy()
            predictions += list(frame_scores)
            continue

        if isinstance(recon_error, (list, tuple)):
            ts_scores = recon_error[0].detach().cpu().numpy()
            teacher_scores = recon_error[1].detach().cpu().numpy()
            fused = fuse_ts_teacher_scores(ts_scores, teacher_scores, args)
            predictions += list(fused.detach().cpu().numpy() if torch.is_tensor(fused) else fused)
        else:
            predictions += list(recon_error.detach().cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    return evaluate_model(
        predictions,
        labels,
        videos,
        normalize_scores=getattr(args, "smooth_normalize", False),
        range=getattr(args, "smooth_range", 38 if args.dataset == "avenue" else 900),
        mu=getattr(args, "smooth_mu", 11 if args.dataset == "avenue" else 282),
        temporal_peak_window=int(getattr(args, "temporal_peak_window", 1) or 1),
    )


def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False,
                   temporal_peak_window=1):

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = apply_temporal_peak_pooling(pred, temporal_peak_window)
        pred = filt(pred, range=range, mu=mu)
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        pred = np.nan_to_num(pred, nan=0.)

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(
        f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}, "
        f"range:{range}, mu:{mu}, normalize scores:{normalize_scores}"
    )
    return micro_auc, macro_auc
