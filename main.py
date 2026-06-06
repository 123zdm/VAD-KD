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
from engine_train import train_one_epoch, test_one_epoch
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
        dataset_train = AbnormalDatasetGradientsTrain(args)
        print(dataset_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

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
        ts_abnormal_strategy=args.ts_abnormal_strategy,
        ts_margin_lambda=args.ts_margin_lambda,
        ts_loss_type=args.ts_loss_type,
        bw2_eps=args.bw2_eps,
        ts_bw2_alpha=args.ts_bw2_alpha,
        ts_bw2_normalize=args.ts_bw2_normalize,
        ts_bw2_rank=args.ts_bw2_rank,
        ts_gram_lambda=args.ts_gram_lambda,
        ts_gram_max_patches=args.ts_gram_max_patches,
        use_cls_head=args.use_cls_head,
        cls_loss_weight=args.cls_loss_weight,
        use_paper_fusion=args.use_paper_fusion,
        inf_alpha=args.inf_alpha,
        inf_beta=args.inf_beta,
        inf_gamma=args.inf_gamma,
    )
    if args.dataset == 'avenue':
        model = mae_cvt_patch16(**model_kwargs).float()
    else:
        model = mae_cvt_patch8(**model_kwargs).float()
    model.to(device)
    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model)
    elif args.run_type == "inference":
        student = torch.load(
            args.output_dir + "/checkpoint-best-student.pth",
            map_location="cpu",
            weights_only=False,
        )['model']
        teacher = torch.load(
            args.output_dir + "/checkpoint-best.pth",
            map_location="cpu",
            weights_only=False,
        )['model']
        for key in student:
            if 'student' in key:
                teacher[key] = student[key]
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)



def do_training(args, data_loader_test, data_loader_train, device, log_writer, model):
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

        train_stats = train_one_epoch(
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
        '--ts_abnormal_strategy',
        type=str,
        default=None,
        choices=['all', 'skip', 'margin'],
        help='Stage-2 distillation on abnormal frames: all(E0), skip(E1), margin(E2)',
    )
    parser.add_argument(
        '--ts_loss_type',
        type=str,
        default=None,
        choices=['mse', 'bw2', 'bw2_mse', 'bw2_lowrank', 'bw2_lowrank_mse'],
        help='Stage-2 distillation loss',
    )
    parser.add_argument('--ts_margin_lambda', type=float, default=None)
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
    parser.add_argument(
        '--masking_method',
        type=str,
        default=None,
        choices=['random_masking', 'grad_masking_v1'],
    )
    parser.add_argument(
        '--use_cls_head',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable cls head (Stage-1 BCE on synthetic abnormal frames)',
    )
    parser.add_argument(
        '--use_paper_fusion',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Use paper pixel-level score fusion (Eq.6, includes cls when enabled)',
    )
    parser.add_argument('--cls_loss_weight', type=float, default=None)
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
    cli_args = parser.parse_args()
    if cli_args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()
    if cli_args.ts_abnormal_strategy is not None:
        args.ts_abnormal_strategy = cli_args.ts_abnormal_strategy
    if cli_args.ts_loss_type is not None:
        args.ts_loss_type = cli_args.ts_loss_type
    if cli_args.ts_margin_lambda is not None:
        args.ts_margin_lambda = cli_args.ts_margin_lambda
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
    if cli_args.masking_method is not None:
        args.masking_method = cli_args.masking_method
    if cli_args.use_cls_head is not None:
        args.use_cls_head = cli_args.use_cls_head
    if cli_args.use_paper_fusion is not None:
        args.use_paper_fusion = cli_args.use_paper_fusion
    if cli_args.cls_loss_weight is not None:
        args.cls_loss_weight = cli_args.cls_loss_weight
    if cli_args.student_only:
        args.student_only = True
    if cli_args.teacher_checkpoint is not None:
        args.teacher_checkpoint = cli_args.teacher_checkpoint
    if args.student_only and not args.teacher_checkpoint:
        raise ValueError("--student_only requires --teacher_checkpoint")
    if not 0.0 <= args.ts_bw2_alpha <= 1.0:
        raise ValueError(f"--ts_bw2_alpha must be in [0, 1], got {args.ts_bw2_alpha}")
    if cli_args.run_type is not None:
        args.run_type = cli_args.run_type
    if cli_args.experiment_name is not None:
        args.experiment_name = cli_args.experiment_name
    else:
        if args.ts_loss_type in ('bw2_mse', 'bw2_lowrank_mse'):
            alpha_tag = int(round(args.ts_bw2_alpha * 100))
            norm_tag = 'norm' if args.ts_bw2_normalize else 'raw'
            if args.ts_loss_type == 'bw2_lowrank_mse':
                loss_prefix = f'bw2lr_r{args.ts_bw2_rank}_a{alpha_tag:02d}_{norm_tag}'
            else:
                loss_prefix = f'bw2mse_a{alpha_tag:02d}_{norm_tag}'
            if args.ts_gram_lambda > 0:
                gram_tag = int(round(args.ts_gram_lambda * 100))
                loss_prefix = f'{loss_prefix}_g{gram_tag:02d}'
        elif args.ts_loss_type in ('bw2', 'bw2_lowrank'):
            loss_prefix = 'bw2lr' if args.ts_loss_type == 'bw2_lowrank' else 'bw2'
        else:
            loss_prefix = 'mse'
        strategy_to_name = {
            'all': f'{loss_prefix}_all',
            'skip': f'{loss_prefix}_skip',
            'margin': f'{loss_prefix}_margin',
        }
        args.experiment_name = strategy_to_name.get(args.ts_abnormal_strategy, f'{loss_prefix}_custom')
    args.output_dir = os.path.join(
        "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae/output",
        args.dataset,
        args.experiment_name,
    )
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
