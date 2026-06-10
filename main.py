import argparse
import datetime
import json
import os
import shutil
import time
from pathlib import Path

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from data.clip_train_dataset import ClipDatasetGradientsTrain
from engine_train import train_one_epoch, test_one_epoch
from engine_train_clip import train_one_epoch_clip
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc
import torch

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    log_writer = SummaryWriter(log_dir=args.output_dir)

    device = args.device
    if args.run_type =='train':
        clip_len = int(getattr(args, "clip_len", 1))
        if clip_len > 1:
            dataset_train = ClipDatasetGradientsTrain(args)
            train_one_epoch_fn = train_one_epoch_clip
        else:
            dataset_train = AbnormalDatasetGradientsTrain(args)
            train_one_epoch_fn = train_one_epoch
        print(dataset_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        train_one_epoch_fn = train_one_epoch

    dataset_test = AbnormalDatasetGradientsTest(args)
    print(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # define the model
    model_kwargs = dict(
        norm_pix_loss=args.norm_pix_loss,
        img_size=args.input_size,
        use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
        abnormal_score_func=args.abnormal_score_func,
        masking_method=args.masking_method,
        grad_weighted_loss=args.grad_weighted_rec_loss,
        ts_loss_type=args.ts_loss_type,
        bw2_eps=args.bw2_eps,
        ts_bw2_alpha=args.ts_bw2_alpha,
        ts_bw2_normalize=args.ts_bw2_normalize,
        ts_bw2_rank=args.ts_bw2_rank,
        ts_gram_lambda=args.ts_gram_lambda,
        ts_gram_max_patches=args.ts_gram_max_patches,
        ts_joint_lambda=args.ts_joint_lambda,
        ts_joint_rank=args.ts_joint_rank,
        ts_joint_stat=args.ts_joint_stat,
        ts_contrastive_margin=args.ts_contrastive_margin,
        ts_contrastive_lambda=args.ts_contrastive_lambda,
        use_anomaly_map_loss=args.use_anomaly_map_loss,
        anomaly_map_loss_weight=args.anomaly_map_loss_weight,
        use_fg_gated_distill=args.use_fg_gated_distill,
        fg_grad_threshold=args.fg_grad_threshold,
        fg_map_threshold=args.fg_map_threshold,
        use_patch_attn_score=args.use_patch_attn_score,
        patch_attn_loss_weight=args.patch_attn_loss_weight,
        patch_attn_in_dim=args.patch_attn_in_dim,
        patch_attn_hidden=args.patch_attn_hidden,
        use_topk_patch_score=getattr(args, 'use_topk_patch_score', False),
        topk_patch_k=getattr(args, 'topk_patch_k', 8),
        use_hard_normal_mining=getattr(args, 'use_hard_normal_mining', False),
        hard_normal_grad_threshold=getattr(args, 'hard_normal_grad_threshold', 0.35),
        hard_normal_loss_weight=getattr(args, 'hard_normal_loss_weight', 0.5),
        use_map_infer_score=getattr(args, 'use_map_infer_score', False),
        map_infer_weight=getattr(args, 'map_infer_weight', 0.3),
        score_weight_teacher=getattr(args, 'score_weight_teacher', 0.4),
        score_weight_ts=getattr(args, 'score_weight_ts', 0.3),
    )
    if args.dataset == 'avenue':
        model = mae_cvt_patch16(**model_kwargs).float()
    else:
        model = mae_cvt_patch8(**model_kwargs).float()
    model.to(device)
    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model, train_one_epoch_fn)
    elif args.run_type == "inference":
        from util.visualize_paper_fig4 import remap_checkpoint_keys

        teacher_ckpt = getattr(args, "teacher_checkpoint", "") or os.path.join(
            args.output_dir, "checkpoint-best.pth"
        )
        student_ckpt = getattr(args, "student_checkpoint", "") or os.path.join(
            args.output_dir, "checkpoint-best-student.pth"
        )
        teacher = remap_checkpoint_keys(
            torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"]
        )
        student = remap_checkpoint_keys(
            torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"]
        )
        for key in student:
            if "student" in key:
                teacher[key] = student[key]
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)



def do_training(args, data_loader_test, data_loader_train, device, log_writer, model, train_one_epoch_fn):
    print("actual lr: %.2e" % args.lr)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    if getattr(args, 'optimizer', 'adamw').lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.student_only:
        latest_ckpt = os.path.join(args.output_dir, "checkpoint-latest.pth")
        if args.resume and os.path.isfile(latest_ckpt):
            checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
            if missing:
                print(f"Warning: missing keys when resuming: {len(missing)}")
            if unexpected:
                print(f"Warning: unexpected keys when resuming: {len(unexpected)}")
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            args.start_epoch = int(checkpoint.get('epoch', args.start_TS_epoch - 1)) + 1
            print(
                f"Resumed from {latest_ckpt}; "
                f"continue from epoch {args.start_epoch}/{args.epochs - 1}"
            )
        else:
            misc.load_teacher_checkpoint(args, model)
            teacher_ckpt_dst = os.path.join(args.output_dir, "checkpoint-best.pth")
            if os.path.abspath(args.teacher_checkpoint) != os.path.abspath(teacher_ckpt_dst):
                shutil.copy2(args.teacher_checkpoint, teacher_ckpt_dst)
                print(f"Copied teacher checkpoint to {teacher_ckpt_dst}")
    else:
        misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs (from epoch {args.start_epoch})")
    start_time = time.time()
    best_micro = 0.0
    best_micro_student = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch_fn(
            model, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args
        )
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        test_stats = test_one_epoch(
            model, data_loader_test, device, epoch, log_writer=log_writer, args=args
        )
        log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

        if args.output_dir:
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, latest=True)
        if not args.student_only and test_stats['micro'] > best_micro:
            best_micro = test_stats['micro']
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, best=True)
        if args.start_TS_epoch <= epoch:
            if test_stats['micro'] > best_micro_student:
                best_micro_student = test_stats['micro']
                misc.save_model(args=args, model=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best=True, student=True)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_test) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    parser.add_argument(
        '--ts_loss_type',
        type=str,
        default=None,
        choices=['mse', 'mse_tw', 'bw2', 'bw2_mse', 'bw2_mse_tw', 'bw2_lowrank', 'bw2_lowrank_mse',
                 'contrastive',
                 'temporal_joint', 'temporal_joint_mse',
                 'temporal_joint_bw2mse', 'temporal_joint_bw2lr_mse'],
        help='Stage-2 distillation loss',
    )
    parser.add_argument('--bw2_eps', type=float, default=None)
    parser.add_argument(
        '--ts_bw2_alpha',
        type=float,
        default=None,
        help='Weight on BW2 in hybrid loss after optional normalization (default 0.5)',
    )
    parser.add_argument(
        '--ts_bw2_normalize',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Normalize MSE/BW2 to batch mean ~1 before hybrid mix (default: True)',
    )
    parser.add_argument('--ts_bw2_rank', type=int, default=None, help='Low-rank BW2 subspace dim')
    parser.add_argument(
        '--ts_gram_lambda',
        type=float,
        default=None,
        help='Optional patch Gram alignment weight (0 disables)',
    )
    parser.add_argument('--ts_gram_max_patches', type=int, default=None)
    parser.add_argument('--clip_len', type=int, default=None, help='K frames per clip (1=single-frame)')
    parser.add_argument('--clip_stride', type=int, default=None, help='Stride when building clip index')
    parser.add_argument('--ts_joint_lambda', type=float, default=None, help='Weight on clip joint BW2 in temporal_joint_mse')
    parser.add_argument('--ts_joint_rank', type=int, default=None, help='Low-rank dim for temporal joint BW2')
    parser.add_argument(
        '--ts_joint_stat',
        type=str,
        default=None,
        choices=['mean', 'mean_std'],
        help='Per-frame stat for temporal joint BW2',
    )
    parser.add_argument(
        '--ts_contrastive_margin',
        type=float,
        default=None,
        help='Min gap (abnormal-normal) for contrastive distillation loss (default 0.005)',
    )
    parser.add_argument(
        '--ts_contrastive_lambda',
        type=float,
        default=None,
        help='Weight on the push term in contrastive distillation loss (default 1.0)',
    )
    parser.add_argument(
        '--masking_method',
        type=str,
        default=None,
        choices=['random_masking', 'grad_masking_v1'],
    )
    parser.add_argument(
        '--use_anomaly_map_loss',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-A: pixel-level anomaly map BCE in Stage-1',
    )
    parser.add_argument('--anomaly_map_loss_weight', type=float, default=None)
    parser.add_argument(
        '--use_fg_gated_distill',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-A: distill only on background patches in Stage-2',
    )
    parser.add_argument('--fg_grad_threshold', type=float, default=None)
    parser.add_argument('--fg_map_threshold', type=float, default=None)
    parser.add_argument(
        '--use_patch_attn_score',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-A: learnable patch-attention frame score readout',
    )
    parser.add_argument('--patch_attn_loss_weight', type=float, default=None)
    parser.add_argument('--patch_attn_in_dim', type=int, default=None)
    parser.add_argument('--patch_attn_hidden', type=int, default=None)
    parser.add_argument(
        '--use_topk_patch_score',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-B: top-k patch aggregation for frame score (FN fix)',
    )
    parser.add_argument('--topk_patch_k', type=int, default=None)
    parser.add_argument(
        '--temporal_peak_window',
        type=int,
        default=None,
        help='Route-B: temporal max-pooling window at inference (1=off)',
    )
    parser.add_argument(
        '--use_hard_normal_mining',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-B: suppress scores on high-motion normal frames (FP fix)',
    )
    parser.add_argument('--hard_normal_grad_threshold', type=float, default=None)
    parser.add_argument('--hard_normal_loss_weight', type=float, default=None)
    parser.add_argument(
        '--use_map_infer_score',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Route-B: fuse decoder ch-4 anomaly map at inference',
    )
    parser.add_argument('--map_infer_weight', type=float, default=None)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size (clip mode: effective frames ≈ batch_size * clip_len per step)',
    )
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_type', type=str, default=None, choices=['train', 'inference'])
    parser.add_argument(
        '--student_only',
        action='store_true',
        help='Load teacher checkpoint and train Stage-2 (student) only',
    )
    parser.add_argument(
        '--teacher_checkpoint',
        type=str,
        default=None,
        help='Path to teacher checkpoint-best.pth (required with --student_only)',
    )
    parser.add_argument(
        '--student_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint-best-student.pth (inference / student-only eval)',
    )
    parser.add_argument(
        '--student_infer_only',
        action='store_true',
        help='Inference with student decoder only; score = student reconstruction error',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from output_dir/checkpoint-latest.pth (Stage-2 student training)',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='DataLoader workers (use 0 if /dev/shm is small)',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default=None,
        choices=['adam', 'adamw'],
        help='Optimizer (paper uses adam)',
    )
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--start_TS_epoch', type=int, default=None)
    parser.add_argument('--percent_abnormal', type=float, default=None)
    parser.add_argument(
        '--paper_baseline',
        action='store_true',
        help='Apply CVPR 2024 paper training hyper-parameters (Adam, two-stage)',
    )
    cli_args = parser.parse_args()
    if cli_args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()
    if cli_args.paper_baseline:
        from configs.configs import apply_paper_baseline
        apply_paper_baseline(args)
    if cli_args.optimizer is not None:
        args.optimizer = cli_args.optimizer
    if cli_args.weight_decay is not None:
        args.weight_decay = cli_args.weight_decay
    if cli_args.lr is not None:
        args.lr = cli_args.lr
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.start_TS_epoch is not None:
        args.start_TS_epoch = cli_args.start_TS_epoch
    if cli_args.percent_abnormal is not None:
        args.percent_abnormal = cli_args.percent_abnormal
    if cli_args.ts_loss_type is not None:
        args.ts_loss_type = cli_args.ts_loss_type
    if cli_args.bw2_eps is not None:
        args.bw2_eps = cli_args.bw2_eps
    if cli_args.ts_bw2_alpha is not None:
        args.ts_bw2_alpha = cli_args.ts_bw2_alpha
    if cli_args.ts_bw2_normalize is not None:
        args.ts_bw2_normalize = cli_args.ts_bw2_normalize
    if cli_args.ts_bw2_rank is not None:
        args.ts_bw2_rank = cli_args.ts_bw2_rank
    if cli_args.ts_gram_lambda is not None:
        args.ts_gram_lambda = cli_args.ts_gram_lambda
    if cli_args.ts_gram_max_patches is not None:
        args.ts_gram_max_patches = cli_args.ts_gram_max_patches
    if cli_args.clip_len is not None:
        args.clip_len = cli_args.clip_len
    if cli_args.clip_stride is not None:
        args.clip_stride = cli_args.clip_stride
    if cli_args.ts_joint_lambda is not None:
        args.ts_joint_lambda = cli_args.ts_joint_lambda
    if cli_args.ts_joint_rank is not None:
        args.ts_joint_rank = cli_args.ts_joint_rank
    if cli_args.ts_joint_stat is not None:
        args.ts_joint_stat = cli_args.ts_joint_stat
    if cli_args.ts_contrastive_margin is not None:
        args.ts_contrastive_margin = cli_args.ts_contrastive_margin
    if cli_args.ts_contrastive_lambda is not None:
        args.ts_contrastive_lambda = cli_args.ts_contrastive_lambda
    if cli_args.masking_method is not None:
        args.masking_method = cli_args.masking_method
    if cli_args.use_anomaly_map_loss is not None:
        args.use_anomaly_map_loss = cli_args.use_anomaly_map_loss
    if cli_args.anomaly_map_loss_weight is not None:
        args.anomaly_map_loss_weight = cli_args.anomaly_map_loss_weight
    if cli_args.use_fg_gated_distill is not None:
        args.use_fg_gated_distill = cli_args.use_fg_gated_distill
    if cli_args.fg_grad_threshold is not None:
        args.fg_grad_threshold = cli_args.fg_grad_threshold
    if cli_args.fg_map_threshold is not None:
        args.fg_map_threshold = cli_args.fg_map_threshold
    if cli_args.use_patch_attn_score is not None:
        args.use_patch_attn_score = cli_args.use_patch_attn_score
    if cli_args.patch_attn_loss_weight is not None:
        args.patch_attn_loss_weight = cli_args.patch_attn_loss_weight
    if cli_args.patch_attn_in_dim is not None:
        args.patch_attn_in_dim = cli_args.patch_attn_in_dim
    if cli_args.patch_attn_hidden is not None:
        args.patch_attn_hidden = cli_args.patch_attn_hidden
    if cli_args.use_topk_patch_score is not None:
        args.use_topk_patch_score = cli_args.use_topk_patch_score
    if cli_args.topk_patch_k is not None:
        args.topk_patch_k = cli_args.topk_patch_k
    if cli_args.temporal_peak_window is not None:
        args.temporal_peak_window = cli_args.temporal_peak_window
    if cli_args.use_hard_normal_mining is not None:
        args.use_hard_normal_mining = cli_args.use_hard_normal_mining
    if cli_args.hard_normal_grad_threshold is not None:
        args.hard_normal_grad_threshold = cli_args.hard_normal_grad_threshold
    if cli_args.hard_normal_loss_weight is not None:
        args.hard_normal_loss_weight = cli_args.hard_normal_loss_weight
    if cli_args.use_map_infer_score is not None:
        args.use_map_infer_score = cli_args.use_map_infer_score
    if cli_args.map_infer_weight is not None:
        args.map_infer_weight = cli_args.map_infer_weight
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.student_only:
        args.student_only = True
    if cli_args.teacher_checkpoint is not None:
        args.teacher_checkpoint = cli_args.teacher_checkpoint
    if cli_args.student_checkpoint is not None:
        args.student_checkpoint = cli_args.student_checkpoint
    if cli_args.student_infer_only:
        args.student_infer_only = True
    if cli_args.resume:
        args.resume = True
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    if args.student_only and not args.teacher_checkpoint:
        raise ValueError("--student_only requires --teacher_checkpoint")
    temporal_joint_types = (
        'temporal_joint', 'temporal_joint_mse',
        'temporal_joint_bw2mse', 'temporal_joint_bw2lr_mse',
    )
    if args.ts_loss_type in temporal_joint_types and int(getattr(args, 'clip_len', 1)) < 2:
        raise ValueError(
            f"--ts_loss_type {args.ts_loss_type} requires --clip_len >= 2"
        )
    if int(getattr(args, 'clip_len', 1)) > 1:
        effective_frames = args.batch_size * args.clip_len
        print(
            f"Clip training: batch_size={args.batch_size}, clip_len={args.clip_len}, "
            f"effective_frames/step≈{effective_frames}"
        )
        if args.ts_loss_type in temporal_joint_types and args.batch_size > 8:
            print(
                f"Warning: temporal_joint* with batch_size={args.batch_size} may OOM "
                f"(K={args.clip_len} forwards kept in graph). Try --batch_size 4~8."
            )
        elif args.batch_size * args.clip_len > 128:
            print(
                f"Warning: batch_size*clip_len={effective_frames} is large; "
                f"try --batch_size {max(100 // args.clip_len, 4)} for ~100 frames/step."
            )
    if not 0.0 <= args.ts_bw2_alpha <= 1.0:
        raise ValueError(f"--ts_bw2_alpha must be in [0, 1], got {args.ts_bw2_alpha}")
    if cli_args.run_type is not None:
        args.run_type = cli_args.run_type
    if cli_args.experiment_name is not None:
        args.experiment_name = cli_args.experiment_name
    else:
        spa_flags = []
        if getattr(args, 'use_anomaly_map_loss', False):
            spa_flags.append('map')
        if getattr(args, 'use_fg_gated_distill', False):
            spa_flags.append('fg')
        if getattr(args, 'use_patch_attn_score', False):
            spa_flags.append('attn')
        if spa_flags:
            args.experiment_name = 'spa_a_' + '_'.join(spa_flags)
        else:
            b_flags = []
            peak_win = int(getattr(args, 'temporal_peak_window', 1) or 1)
            if peak_win > 1:
                b_flags.append(f'peak{peak_win}')
            if getattr(args, 'use_topk_patch_score', False):
                b_flags.append(f'topk{getattr(args, "topk_patch_k", 8)}')
            if getattr(args, 'use_hard_normal_mining', False):
                b_flags.append('hn')
            if getattr(args, 'use_map_infer_score', False):
                b_flags.append('mapinf')
            if b_flags:
                args.experiment_name = 'spa_b_' + '_'.join(b_flags)
            elif args.ts_loss_type in ('bw2_mse', 'bw2_lowrank_mse'):
                alpha_tag = int(round(args.ts_bw2_alpha * 100))
                norm_tag = 'norm' if args.ts_bw2_normalize else 'raw'
                if args.ts_loss_type == 'bw2_lowrank_mse':
                    loss_prefix = f'bw2lr_r{args.ts_bw2_rank}_a{alpha_tag:02d}_{norm_tag}'
                else:
                    loss_prefix = f'bw2mse_a{alpha_tag:02d}_{norm_tag}'
                if args.ts_gram_lambda > 0:
                    gram_tag = int(round(args.ts_gram_lambda * 100))
                    loss_prefix = f'{loss_prefix}_g{gram_tag:02d}'
                args.experiment_name = loss_prefix
            elif args.ts_loss_type in ('bw2', 'bw2_lowrank'):
                args.experiment_name = 'bw2lr' if args.ts_loss_type == 'bw2_lowrank' else 'bw2'
            elif args.ts_loss_type in temporal_joint_types:
                joint_tag = int(round(args.ts_joint_lambda * 100))
                clip_tag = int(getattr(args, 'clip_len', 1))
                if args.ts_loss_type == 'temporal_joint':
                    loss_prefix = f'tjointonly_k{clip_tag}'
                elif args.ts_loss_type in ('temporal_joint_bw2mse', 'temporal_joint_bw2lr_mse'):
                    alpha_tag = int(round(args.ts_bw2_alpha * 100))
                    bw2_tag = 'bw2lr' if args.ts_loss_type == 'temporal_joint_bw2lr_mse' else 'bw2mse'
                    loss_prefix = f'tjoint_{bw2_tag}_a{alpha_tag:02d}_k{clip_tag}_l{joint_tag:02d}'
                else:
                    loss_prefix = f'tjoint_k{clip_tag}_l{joint_tag:02d}'
                args.experiment_name = loss_prefix
            else:
                args.experiment_name = 'mse'
    args.output_dir = os.path.join(
        "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae/output",
        args.dataset,
        args.experiment_name,
    )
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
