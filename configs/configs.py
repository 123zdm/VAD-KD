import ml_collections
import os


_BASE_OUTPUT = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae/output"


def _build_output_dir(dataset, experiment_name):
    return os.path.join(_BASE_OUTPUT, dataset, experiment_name)


def _baseline_defaults(config):
    """Defaults aligned with official repo + successful local ablations."""
    config.epochs = 140
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.optimizer = "adamw"
    config.weight_decay = 0.05

    config.score_weight_teacher = 0.4
    config.score_weight_ts = 0.3

    config.smooth_range = 38 if config.get("dataset", "avenue") == "avenue" else 900
    config.smooth_mu = 11 if config.get("dataset", "avenue") == "avenue" else 282
    config.smooth_normalize = config.get("dataset", "avenue") != "avenue"

    # Inference: run student decoder only (no teacher decoder forward).
    config.student_infer_only = False

    # Route-A spatial-aware extensions (default off for backward compatibility).
    config.use_anomaly_map_loss = False
    config.anomaly_map_loss_weight = 0.5
    config.use_fg_gated_distill = False
    config.fg_grad_threshold = 0.35
    config.fg_map_threshold = 0.1
    config.use_patch_attn_score = False
    config.patch_attn_loss_weight = 0.5
    config.patch_attn_in_dim = 3
    config.patch_attn_hidden = 64

    # Route-B: targeted fixes for hard Avenue videos (default off).
    config.use_topk_patch_score = False
    config.topk_patch_k = 8
    config.temporal_peak_window = 1
    config.use_hard_normal_mining = False
    config.hard_normal_grad_threshold = 0.35
    config.hard_normal_loss_weight = 0.5
    config.use_map_infer_score = False
    config.map_infer_weight = 0.3


def apply_paper_baseline(config):
    """CVPR 2024 paper training hyper-parameters (Official fusion at eval)."""
    config.epochs = 140
    config.start_TS_epoch = 100
    config.optimizer = "adam"
    config.weight_decay = 0.0
    config.lr = 1e-4
    config.batch_size = 100
    config.percent_abnormal = 0.25
    config.grad_weighted_rec_loss = True
    config.masking_method = "grad_masking_v1"
    config.ts_loss_type = "mse"
    config.score_weight_teacher = 0.4
    config.score_weight_ts = 0.3
    config.student_only = False
    config.student_infer_only = False


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.mask_ratio = 0.5
    config.experiment_name = "mse_all"
    config.output_dir = _build_output_dir("avenue", config.experiment_name)
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = False

    # Stage-2 distillation
    config.ts_loss_type = "mse"
    config.bw2_eps = 1e-4
    config.ts_bw2_alpha = 0.5
    config.ts_bw2_normalize = True
    config.ts_bw2_rank = 32
    config.ts_gram_lambda = 0.0
    config.ts_gram_max_patches = 128

    # Clip-level Temporal Joint BW² (separate from single-frame ts_loss_type)
    config.clip_len = 1
    config.clip_stride = 1
    config.ts_joint_lambda = 0.5
    config.ts_joint_rank = 32
    config.ts_joint_stat = "mean"

    # Contrastive distillation
    config.ts_contrastive_margin = 0.005
    config.ts_contrastive_lambda = 1.0

    config.student_only = False
    config.teacher_checkpoint = ""
    config.lr = 1e-4

    config.dataset = "avenue"
    _baseline_defaults(config)

    config.avenue_path = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/hstforu-kd/data/avenue"
    config.avenue_gt_path = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/hstforu-kd/data/avenue/Avenue_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.mask_ratio = 0.5
    config.experiment_name = "mse_all"
    config.output_dir = _build_output_dir("shanghai", config.experiment_name)
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "train"
    config.resume = False

    config.ts_loss_type = "mse"
    config.bw2_eps = 1e-4
    config.ts_bw2_alpha = 0.5
    config.ts_bw2_normalize = True
    config.ts_bw2_rank = 32
    config.ts_gram_lambda = 0.0
    config.ts_gram_max_patches = 128

    # Clip-level Temporal Joint BW² (separate from single-frame ts_loss_type)
    config.clip_len = 1
    config.clip_stride = 1
    config.ts_joint_lambda = 0.5
    config.ts_joint_rank = 32
    config.ts_joint_stat = "mean"

    # Contrastive distillation
    config.ts_contrastive_margin = 0.005
    config.ts_contrastive_lambda = 1.0

    config.student_only = False
    config.teacher_checkpoint = ""
    config.lr = 1e-4

    config.dataset = "shanghai"
    _baseline_defaults(config)

    config.shanghai_path = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/hstforu-kd/data/shanghaitech"
    config.shanghai_gt_path = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/hstforu-kd/data/shanghaitech/Shanghai_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config
