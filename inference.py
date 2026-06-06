from collections.abc import Iterable

import numpy as np
import torch
from sklearn import metrics

from util import misc
from util.abnormal_utils import filt


def fuse_ts_teacher_scores(ts_score, teacher_score, args):
    w_teacher = float(getattr(args, "score_weight_teacher", 0.4))
    w_ts = float(getattr(args, "score_weight_ts", 0.3))
    return w_teacher * teacher_score + w_ts * ts_score


def inference(model: torch.nn.Module, data_loader: Iterable,
              device: torch.device,
              log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions_student_teacher = []
    predictions_teacher = []
    labels = []
    videos = []
    frames = []
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        frames += list(frame_name)
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        model.train_TS = True
        if args.dataset == 'avenue':
            model.abnormal_score_func_TS = "L2"
        else:
            model.abnormal_score_func_TS = 'L1'
        _, _, _, recon_error = model(
            samples, targets=targets, grad_mask=grads, mask_ratio=args.mask_ratio
        )
        if isinstance(recon_error, (list, tuple)):
            predictions_student_teacher += list(recon_error[0].detach().cpu().numpy())
            predictions_teacher += list(recon_error[1].detach().cpu().numpy())
        else:
            frame_scores = recon_error.detach().cpu().numpy()
            predictions_student_teacher += list(frame_scores)
            predictions_teacher += list(np.zeros_like(frame_scores))

    predictions_student_teacher = np.array(predictions_student_teacher)
    predictions_teacher = np.array(predictions_teacher)
    if getattr(args, "use_paper_fusion", False):
        predictions = predictions_student_teacher
    else:
        predictions = fuse_ts_teacher_scores(predictions_student_teacher, predictions_teacher, args)
    labels = np.array(labels)
    videos = np.array(videos)

    return evaluate_model(
        predictions,
        labels,
        videos,
        normalize_scores=getattr(args, "smooth_normalize", False),
        range=getattr(args, "smooth_range", 38 if args.dataset == "avenue" else 900),
        mu=getattr(args, "smooth_mu", 11 if args.dataset == "avenue" else 282),
    )


def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False):

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
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
