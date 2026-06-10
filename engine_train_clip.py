"""Training loop for K-frame clip distillation (separate from engine_train.py)."""

import math
import sys

import torch
import util.misc as misc
from typing import Iterable


def _flatten_clip_batch(samples, grad_mask, targets, is_abnormal):
    """[B, K, ...] -> [B*K, ...] for standard single-frame forward."""
    batch_size, clip_len = samples.shape[:2]
    flat_samples = samples.reshape(batch_size * clip_len, *samples.shape[2:])
    flat_grad = grad_mask.reshape(batch_size * clip_len, *grad_mask.shape[2:])
    flat_targets = targets.reshape(batch_size * clip_len, *targets.shape[2:])
    if is_abnormal.dim() == 2:
        flat_abnormal = is_abnormal.reshape(batch_size * clip_len)
    else:
        flat_abnormal = is_abnormal
    return flat_samples, flat_grad, flat_targets, flat_abnormal, batch_size, clip_len


def train_one_epoch_clip(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
):
    """Clip-aware training; delegates to forward_clip_TS when using temporal joint loss."""
    model.train(True)
    model = model.float()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    if epoch >= args.start_TS_epoch:
        model.train_TS = True
        model.freeze_backbone()
    else:
        model.train_TS = False

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    use_temporal_joint = getattr(model, "uses_temporal_joint_ts", lambda: False)()

    for data_iter_step, (samples, grad_mask, targets, is_abnormal) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        grad_mask = grad_mask.to(device, non_blocking=True)
        is_abnormal = is_abnormal.to(device, non_blocking=True)

        if model.train_TS and use_temporal_joint:
            loss, _, _ = model.forward_clip_TS(
                samples,
                grad_mask=grad_mask,
                targets=targets,
                mask_ratio=args.mask_ratio,
                is_abnormal=is_abnormal,
            )
        else:
            flat_samples, flat_grad, flat_targets, flat_abnormal, _, _ = _flatten_clip_batch(
                samples, grad_mask, targets, is_abnormal
            )
            loss, _, _ = model(
                flat_samples,
                grad_mask=flat_grad,
                targets=flat_targets,
                mask_ratio=args.mask_ratio,
                is_abnormal=flat_abnormal,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
