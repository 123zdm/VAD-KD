import ml_collections
import os


_BASE_OUTPUT = "/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae/output"


def _build_output_dir(dataset, experiment_name):
    return os.path.join(_BASE_OUTPUT, dataset, experiment_name)


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.epochs = 200
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
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
    # ts_loss_type: "mse" | "bw2" | "bw2_mse" (alpha*BW2 + (1-alpha)*MSE)
    # ts_abnormal_strategy: "all" | "skip" | "margin"
    config.ts_loss_type = "mse"
    config.ts_abnormal_strategy = "all"
    config.ts_margin_lambda = 0.1
    config.bw2_eps = 1e-4
    config.ts_bw2_alpha = 0.3

    # Stage-2-only training (load teacher checkpoint, skip Stage 1)
    config.student_only = False
    config.teacher_checkpoint = ""
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
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
    config.epochs = 200
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
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
    config.ts_abnormal_strategy = "all"
    config.ts_margin_lambda = 0.1
    config.bw2_eps = 1e-4
    config.ts_bw2_alpha = 0.3

    # Stage-2-only training (load teacher checkpoint, skip Stage 1)
    config.student_only = False
    config.teacher_checkpoint = ""
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
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
