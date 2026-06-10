import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from model.cvt import ConvEmbed, Block
from util.bw2_loss import (
    diagonal_bw2,
    gram_patch_loss,
    lowrank_bw2,
    normalize_loss_terms,
)
from util.morphology import Erosion2d, Dilation2d
from model.spatial_modules import (
    PatchAttentionScoreHead,
    build_foreground_patch_mask,
    normalize_patch_features,
    pool_map_to_patches,
)


class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(512,512), patch_size=16, in_chans=9, out_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', masking_method="random_masking",
                 grad_weighted_loss=True, student_depth=1,
                 ts_loss_type="mse", bw2_eps=1e-4, ts_bw2_alpha=0.3,
                 ts_bw2_normalize=True, ts_bw2_rank=32, ts_gram_lambda=0.0,
                 ts_gram_max_patches=128,
                 ts_joint_lambda=0.5, ts_joint_rank=32, ts_joint_stat="mean",
                 ts_contrastive_margin=0.005, ts_contrastive_lambda=1.0,
                 use_anomaly_map_loss=False, anomaly_map_loss_weight=0.5,
                 use_fg_gated_distill=False, fg_grad_threshold=0.35, fg_map_threshold=0.1,
                 use_patch_attn_score=False, patch_attn_loss_weight=0.5,
                 patch_attn_in_dim=3, patch_attn_hidden=64,
                 use_topk_patch_score=False, topk_patch_k=8,
                 use_hard_normal_mining=False, hard_normal_grad_threshold=0.35,
                 hard_normal_loss_weight=0.5,
                 use_map_infer_score=False, map_infer_weight=0.3,
                 score_weight_teacher=0.4, score_weight_ts=0.3):
        super().__init__()
        # --------------------------------------------------------------------------
        # Abnormal specifics
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        if isinstance(abnormal_score_func, (list, tuple)) and len(abnormal_score_func) >= 2:
            self.abnormal_score_func = abnormal_score_func[0]
            self.abnormal_score_func_TS = abnormal_score_func[1]
        elif isinstance(abnormal_score_func, str):
            self.abnormal_score_func = abnormal_score_func
            self.abnormal_score_func_TS = abnormal_score_func
        else:
            raise ValueError("abnormal_score_func must be a str or a sequence of at least 2 elements")
        # --------------------------------------------------------------------------

        self.masking = getattr(self, masking_method)
        self.grad_weighted_loss=grad_weighted_loss
        self.ts_loss_type = ts_loss_type
        self.bw2_eps = bw2_eps
        self.ts_bw2_alpha = ts_bw2_alpha
        self.ts_bw2_normalize = ts_bw2_normalize
        self.ts_bw2_rank = ts_bw2_rank
        self.ts_gram_lambda = ts_gram_lambda
        self.ts_gram_max_patches = ts_gram_max_patches
        self.ts_joint_lambda = ts_joint_lambda
        self.ts_joint_rank = ts_joint_rank
        self.ts_joint_stat = ts_joint_stat
        self.ts_contrastive_margin = ts_contrastive_margin
        self.ts_contrastive_lambda = ts_contrastive_lambda
        self.use_anomaly_map_loss = use_anomaly_map_loss
        self.anomaly_map_loss_weight = anomaly_map_loss_weight
        self.use_fg_gated_distill = use_fg_gated_distill
        self.fg_grad_threshold = fg_grad_threshold
        self.fg_map_threshold = fg_map_threshold
        self.use_patch_attn_score = use_patch_attn_score
        self.patch_attn_loss_weight = patch_attn_loss_weight
        self.use_topk_patch_score = use_topk_patch_score
        self.topk_patch_k = topk_patch_k
        self.use_hard_normal_mining = use_hard_normal_mining
        self.hard_normal_grad_threshold = hard_normal_grad_threshold
        self.hard_normal_loss_weight = hard_normal_loss_weight
        self.use_map_infer_score = use_map_infer_score
        self.map_infer_weight = map_infer_weight
        self.score_weight_teacher = score_weight_teacher
        self.score_weight_ts = score_weight_ts

        assert 0 < student_depth < decoder_depth
        self.student_depth = student_depth
        self.train_TS = False
        self.student_infer_only = False
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_size,
            padding=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.num_patches = img_size[0]//patch_size*img_size[1]//patch_size
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch

        self.decoder_student_block = Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        self.decoder_student_norm = norm_layer(decoder_embed_dim)
        self.decoder_student_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch
        self.out_chans=out_chans
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.erosion = Erosion2d(1, 1, 2, soft_max=False)
        self.dilation = Dilation2d(1, 1, 3, soft_max=False)

        self.erosion_3 = Erosion2d(3, 3, 2, soft_max=False)
        self.dilation_3 = Dilation2d(3, 3, 3, soft_max=False)

        self.patch_score_head = None
        if self.use_patch_attn_score:
            self.patch_score_head = PatchAttentionScoreHead(
                in_dim=patch_attn_in_dim,
                hidden=patch_attn_hidden,
            )

    def freeze_backbone(self):
        self.cls_token.requires_grad = False
        self.mask_token.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.decoder_norm.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_pred.parameters():
            param.requires_grad = False
        for i in range(0, len(self.decoder_blocks)):
            for param in self.decoder_blocks[i].parameters():
                param.requires_grad = False
        if self.patch_score_head is not None:
            for param in self.patch_score_head.parameters():
                param.requires_grad = True

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = self.H
        w = self.W
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_restore

    def grad_masking_v1(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        grad_mask = F.max_pool2d(grad_mask, self.patch_size).max(1).values
        grad_mask = rearrange(grad_mask, 'b h w -> b (h w)')

        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        ids_shuffle = torch.argsort(grad_mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_TS(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Student Transformer blocks
        for idx in range(0, self.student_depth):
            x = self.decoder_blocks[idx](x, self.H, self.W)
        x_student = self.decoder_student_block(x, self.H, self.W)
        x_student = self.decoder_student_norm(x_student)
        x_student = self.decoder_student_pred(x_student)
        x_student = x_student[:, 1:, :]

        for idx in range(self.student_depth, len(self.decoder_blocks)):
            x = self.decoder_blocks[idx](x, self.H, self.W)

        # predictor projection
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x_student, x

    def forward_decoder_student_only(self, x, ids_restore):
        """Student decoder path only — skips remaining teacher decoder blocks."""
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        for idx in range(0, self.student_depth):
            x = self.decoder_blocks[idx](x, self.H, self.W)
        x = self.decoder_student_block(x, self.H, self.W)
        x = self.decoder_student_norm(x)
        x = self.decoder_student_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, gradients, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        min_magnitude_anomaly = torch.ones((gradients.shape[0],1,1,1), device=imgs.device) * 128
        if self.grad_weighted_loss:
            anomaly_map = imgs[:, 3:, :, :]
            anomaly_map = torch.clip(anomaly_map, min=0, max=1)
            anomaly_map *= torch.maximum(min_magnitude_anomaly, torch.amax(gradients, dim=(1, 2, 3), keepdim=True))
            gradients += anomaly_map
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            # grad_weights = (grad_weights - torch.amin(grad_weights, keepdim=True)) / \
            #                (torch.amax(grad_weights, keepdim=True) - torch.amin(grad_weights, keepdim=True))
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def _anomaly_map_loss(self, pred, targets):
        """Pixel-level BCE on decoder channel-4 vs GT mask in target."""
        pred_img = self.unpatchify(pred)
        pred_map = pred_img[:, 3:4]
        gt_map = targets[:, 3:4]
        gt_map = ((gt_map + 1.0) * 0.5).clamp(0.0, 1.0)
        return F.binary_cross_entropy_with_logits(pred_map, gt_map)

    def _patch_score_features(self, targets, pred_stud, pred_teacher):
        """Build per-patch features: teacher recon, ts gap, predicted anomaly map."""
        pred_teacher_img = self.unpatchify(pred_teacher)
        pred_stud_img = self.unpatchify(pred_stud)
        target_rgb = targets[:, :3]

        teacher_map = ((target_rgb - pred_teacher_img[:, :3]) ** 2).mean(dim=1, keepdim=True)
        ts_map = ((pred_teacher_img[:, :3] - pred_stud_img[:, :3]) ** 2).mean(dim=1, keepdim=True)
        map_pred = torch.sigmoid(pred_teacher_img[:, 3:4])

        teacher_p = pool_map_to_patches(teacher_map, self.patch_size)
        ts_p = pool_map_to_patches(ts_map, self.patch_size)
        map_p = pool_map_to_patches(map_pred, self.patch_size)
        feats = torch.stack([teacher_p, ts_p, map_p], dim=-1)
        return normalize_patch_features(feats)

    def _patch_attn_scores(self, targets, pred_stud, pred_teacher):
        feats = self._patch_score_features(targets, pred_stud, pred_teacher)
        readout, frame_logit, attn = self.patch_score_head(feats)
        return readout, frame_logit, attn

    def _ts_patch_mse(self, preds_stud, preds_teacher, mask):
        loss = (preds_stud - preds_teacher) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        denom = mask.sum(dim=1).clamp(min=1)
        return (loss * mask).sum(dim=1) / denom

    def _ts_patch_mse_weighted(self, preds_stud, preds_teacher, target_patches, mask):
        """Patch-level distillation weighted by teacher's own recon difficulty.

        Teacher reconstruction error per patch (stop-grad) is used as a soft
        importance weight: patches the teacher finds "hard" (high recon error)
        carry more weight, focusing distillation on structurally rich regions.

        target_patches: [B, L, D]  — patchified GT (same space as preds)
        """
        patch_loss = ((preds_stud - preds_teacher) ** 2).mean(dim=-1)  # [B, L]
        with torch.no_grad():
            teacher_err = ((preds_teacher - target_patches) ** 2).mean(dim=-1)  # [B, L]
            # only weight within masked (=distilled) patches; add eps for numerical safety
            w = teacher_err * mask + 1e-8
            w = w / w.sum(dim=1, keepdim=True)  # softmax-like, sums to 1 per sample
        return (patch_loss * w * mask).sum(dim=1)  # [B]

    def _uses_lowrank_bw2(self):
        return self.ts_loss_type in (
            "bw2_lowrank",
            "bw2_lowrank_mse",
            "temporal_joint_bw2lr_mse",
        )

    def _bw2_on_patch_pair(self, z_s, z_t):
        if self._uses_lowrank_bw2():
            return lowrank_bw2(z_s, z_t, rank=self.ts_bw2_rank, eps=self.bw2_eps)
        return diagonal_bw2(z_s, z_t, eps=self.bw2_eps)

    def _ts_distill_per_sample(self, preds_stud, preds_teacher, mask):
        batch_size = preds_stud.shape[0]
        bw2_losses = []
        gram_losses = []
        for b in range(batch_size):
            idx = mask[b].bool()
            if idx.sum() == 0:
                bw2_losses.append(preds_stud.new_zeros(()))
                gram_losses.append(preds_stud.new_zeros(()))
                continue
            z_s = preds_stud[b, idx]
            z_t = preds_teacher[b, idx]
            bw2_losses.append(self._bw2_on_patch_pair(z_s, z_t))
            if self.ts_gram_lambda > 0.0:
                gram_losses.append(
                    gram_patch_loss(z_s, z_t, max_patches=self.ts_gram_max_patches)
                )
        bw2_out = torch.stack(bw2_losses)
        if self.ts_gram_lambda > 0.0:
            return bw2_out, torch.stack(gram_losses)
        return bw2_out, None

    def _combine_bw2_mse(self, per_sample_mse, per_sample_bw2):
        alpha = self.ts_bw2_alpha
        if self.ts_bw2_normalize:
            per_sample_mse, per_sample_bw2 = normalize_loss_terms(
                per_sample_mse, per_sample_bw2
            )
        return (1.0 - alpha) * per_sample_mse + alpha * per_sample_bw2

    def forward_loss_TS(self, preds_stud, preds_teacher, mask, target_patches=None,
                        grad_mask=None, targets=None):
        distill_mask = mask
        if self.use_fg_gated_distill and grad_mask is not None and targets is not None:
            fg = build_foreground_patch_mask(
                grad_mask,
                targets,
                self.patch_size,
                grad_threshold=self.fg_grad_threshold,
                map_threshold=self.fg_map_threshold,
            )
            distill_mask = mask * (1.0 - fg)

        # --- uniform MSE (original) ---
        per_sample_mse = self._ts_patch_mse(preds_stud, preds_teacher, distill_mask)

        if self.ts_loss_type == "mse":
            per_sample = per_sample_mse

        # --- teacher-recon weighted MSE (direction 2) ---
        elif self.ts_loss_type == "mse_tw":
            if target_patches is None:
                per_sample = per_sample_mse  # fallback at test time
            else:
                per_sample = self._ts_patch_mse_weighted(
                    preds_stud, preds_teacher, target_patches, distill_mask
                )

        # --- weighted BW²+MSE (direction 2 × BW²) ---
        elif self.ts_loss_type == "bw2_mse_tw":
            per_sample_bw2, per_sample_gram = self._ts_distill_per_sample(
                preds_stud, preds_teacher, mask
            )
            if target_patches is not None:
                base_mse = self._ts_patch_mse_weighted(preds_stud, preds_teacher, target_patches, mask)
            else:
                base_mse = per_sample_mse
            per_sample = self._combine_bw2_mse(base_mse, per_sample_bw2)
            if per_sample_gram is not None:
                if self.ts_bw2_normalize:
                    (per_sample_gram,) = normalize_loss_terms(per_sample_gram)
                per_sample = per_sample + self.ts_gram_lambda * per_sample_gram

        # --- contrastive distillation (direction 4) ---
        elif self.ts_loss_type == "contrastive":
            # per_sample_mse is the gap dist(stud, teacher) per frame [B]
            # target: normal frames → small gap; abnormal frames → large gap
            # stored on model so forward() can pass is_abnormal; handled below
            per_sample = per_sample_mse  # real contrastive logic applied in forward()


        elif self.ts_loss_type in ("bw2", "bw2_lowrank"):
            per_sample_bw2, per_sample_gram = self._ts_distill_per_sample(
                preds_stud, preds_teacher, mask
            )
            per_sample = per_sample_bw2
            if per_sample_gram is not None:
                per_sample = per_sample + self.ts_gram_lambda * per_sample_gram
        elif self.ts_loss_type in ("bw2_mse", "bw2_lowrank_mse"):
            per_sample_bw2, per_sample_gram = self._ts_distill_per_sample(
                preds_stud, preds_teacher, mask
            )
            per_sample = self._combine_bw2_mse(per_sample_mse, per_sample_bw2)
            if per_sample_gram is not None:
                if self.ts_bw2_normalize:
                    (per_sample_gram,) = normalize_loss_terms(per_sample_gram)
                per_sample = per_sample + self.ts_gram_lambda * per_sample_gram
        elif self.ts_loss_type in (
            "temporal_joint",
            "temporal_joint_mse",
            "temporal_joint_bw2mse",
            "temporal_joint_bw2lr_mse",
        ):
            # Clip joint loss lives in forward_clip_TS; single-frame eval/train
            # fallback uses the per-frame component only (test_one_epoch path).
            if self.ts_loss_type in ("temporal_joint_bw2mse", "temporal_joint_bw2lr_mse"):
                per_sample_bw2, per_sample_gram = self._ts_distill_per_sample(
                    preds_stud, preds_teacher, mask
                )
                per_sample = self._combine_bw2_mse(per_sample_mse, per_sample_bw2)
                if per_sample_gram is not None:
                    if self.ts_bw2_normalize:
                        (per_sample_gram,) = normalize_loss_terms(per_sample_gram)
                    per_sample = per_sample + self.ts_gram_lambda * per_sample_gram
            else:
                per_sample = per_sample_mse
        else:
            raise ValueError(f"Unknown ts_loss_type: {self.ts_loss_type}")

        return per_sample

    def _contrastive_ts_loss(self, per_sample_gap, is_abnormal):
        """Direction 4: Contrastive distillation.

        Explicitly trains the teacher-student gap to be:
          - small  on normal frames  (standard distillation goal)
          - larger on pseudo-anomaly frames  (test-time anomaly detection goal)

        Loss = mean_gap_normal  +  ReLU(margin − (mean_gap_abnormal − mean_gap_normal))
             = pull normal close  +  push abnormal gap away if it's not already far enough

        This directly optimises the quantity measured at test time (gap score),
        unlike skip (which ignores abnormal) or margin (which subtracts a loss term).
        """
        is_abnormal = is_abnormal.bool()
        normal_mask = ~is_abnormal

        if normal_mask.any():
            loss_normal = per_sample_gap[normal_mask].mean()
        else:
            return per_sample_gap.sum() * 0.0

        if is_abnormal.any():
            gap_normal = per_sample_gap[normal_mask].mean().detach()
            gap_abnormal = per_sample_gap[is_abnormal].mean()
            # push gap_abnormal to be at least (gap_normal + ts_contrastive_margin)
            push_loss = F.relu(self.ts_contrastive_margin - (gap_abnormal - gap_normal))
            loss = loss_normal + self.ts_contrastive_lambda * push_loss
        else:
            loss = loss_normal

        return loss

    def forward(self, imgs, targets, grad_mask=None, mask_ratio=0.75, is_abnormal=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask)

        if not self.training and self.student_infer_only:
            pred_stud = self.forward_decoder_student_only(latent, ids_restore)
            score = self.abnormal_score(targets, pred_stud, mask, grad_mask)
            return pred_stud.new_zeros(()), pred_stud, mask, score

        if self.train_TS is False:
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(targets, grad_mask, pred, mask)
            if self.training and self.use_anomaly_map_loss:
                loss = loss + self.anomaly_map_loss_weight * self._anomaly_map_loss(pred, targets)
            if self.training:
                return loss, pred, mask
            else:
                return loss, pred, mask, self.abnormal_score(targets, pred, mask, grad_mask)
        else:
            pred_stud, pred_teacher = self.forward_decoder_TS(latent, ids_restore)  # [N, L, p*p*3]
            target_patches = self.patchify(targets[:, :self.out_chans])
            per_sample = self.forward_loss_TS(
                pred_stud, pred_teacher, mask,
                target_patches=target_patches if self.training else None,
                grad_mask=grad_mask if self.training else None,
                targets=targets if self.training else None,
            )
            if self.ts_loss_type == "contrastive" and self.training and is_abnormal is not None:
                loss = self._contrastive_ts_loss(per_sample, is_abnormal)
            else:
                loss = per_sample.mean()
            if self.training and self.use_patch_attn_score and is_abnormal is not None:
                _, frame_logit, _ = self._patch_attn_scores(targets, pred_stud, pred_teacher)
                attn_loss = F.binary_cross_entropy_with_logits(
                    frame_logit, is_abnormal.float()
                )
                loss = loss + self.patch_attn_loss_weight * attn_loss
            if (
                self.training
                and self.use_hard_normal_mining
                and is_abnormal is not None
                and grad_mask is not None
            ):
                hn_loss = self._hard_normal_suppression_loss(
                    targets, pred_stud, pred_teacher, mask, grad_mask, is_abnormal
                )
                loss = loss + self.hard_normal_loss_weight * hn_loss
            if self.training:
                return loss, pred_stud, mask
            else:
                return loss, pred_teacher, mask, self.abnormal_score_TS(
                    targets, pred_stud, pred_teacher, mask, grad_mask
                )

    def abnormal_score(self, imgs, pred, mask, gradients):
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(0, imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])

            pred = torch.stack(selected_pred)
            imgs = torch.stack(selected_lbl)
        return ((imgs - pred) ** 2).mean((1, 2))  # MSE

    def _pool_patch_scores(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate per-patch scores to frame level (mean or top-k)."""
        if self.use_topk_patch_score:
            k = min(self.topk_patch_k, patch_scores.shape[1])
            vals, _ = patch_scores.topk(k, dim=1)
            return vals.mean(dim=1)
        return patch_scores.mean(dim=1)

    def _ts_patch_score_tensors(self, imgs, pred_stud, pred_teacher):
        """Return per-patch TS gap and teacher recon errors [B, L]."""
        imgs = self.patchify(imgs)
        if self.abnormal_score_func_TS == "L1":
            ts_patch = torch.abs(pred_teacher - pred_stud).mean(2)
            teacher_patch = torch.abs(imgs - pred_teacher).mean(2)
        elif self.abnormal_score_func_TS == "L2":
            ts_patch = ((pred_teacher - pred_stud) ** 2).mean(2)
            teacher_patch = ((imgs - pred_teacher) ** 2).mean(2)
        else:
            raise ValueError(f"Unsupported TS score func: {self.abnormal_score_func_TS}")
        return ts_patch, teacher_patch

    def _official_fused_frame_scores(self, ts_patch, teacher_patch):
        ts_score = self._pool_patch_scores(ts_patch)
        teacher_score = self._pool_patch_scores(teacher_patch)
        fused = (
            self.score_weight_teacher * teacher_score
            + self.score_weight_ts * ts_score
        )
        return fused, ts_score, teacher_score

    def _map_infer_frame_scores(self, pred_teacher):
        pred_img = self.unpatchify(pred_teacher)
        return torch.sigmoid(pred_img[:, 3:4]).amax(dim=(1, 2, 3))

    def _hard_normal_suppression_loss(
        self, targets, pred_stud, pred_teacher, mask, grad_mask, is_abnormal
    ):
        ts_patch, teacher_patch = self._ts_patch_score_tensors(targets, pred_stud, pred_teacher)
        fused, _, _ = self._official_fused_frame_scores(ts_patch, teacher_patch)
        grad_level = grad_mask.float().mean(dim=(1, 2, 3))
        grad_level = grad_level / (grad_level.amax(dim=0, keepdim=True) + 1e-6)
        hard_normal = (grad_level >= self.hard_normal_grad_threshold) & (~is_abnormal.bool())
        if not hard_normal.any():
            return fused.new_zeros(())
        return fused[hard_normal].mean()

    def abnormal_score_TS(self, imgs, pred_stud, pred_teacher, mask, gradients):
        if self.use_patch_attn_score and self.patch_score_head is not None:
            # Use frame_logit (trained with BCE), not readout (untrained weighted recon).
            _, frame_logit, _ = self._patch_attn_scores(imgs, pred_stud, pred_teacher)
            return torch.sigmoid(frame_logit)

        ts_patch, teacher_patch = self._ts_patch_score_tensors(imgs, pred_stud, pred_teacher)
        fused, ts_score, teacher_score = self._official_fused_frame_scores(ts_patch, teacher_patch)

        if self.use_map_infer_score:
            map_score = self._map_infer_frame_scores(pred_teacher)
            w_map = self.map_infer_weight
            w_base = max(1.0 - w_map, 1e-6)
            fused = w_base * fused + w_map * map_score
            return fused

        return [ts_score, teacher_score]

    def process_result(self, gradients, pred_stud, pred_teacher, do_erosion=True):
        gradients = gradients.mean(dim=1,keepdim=True)
        gradients = (gradients - torch.amin(gradients, dim=(1, 2), keepdim=True)) / (
                    torch.amax(gradients, dim=(1, 2), keepdim=True)
                    - torch.amin(gradients, dim=(1, 2), keepdim=True))

        teacher_student = ((pred_teacher - pred_stud) ** 2)


        if do_erosion:
            teacher_student = self.unpatchify(teacher_student)
            teacher_student *= gradients


            teacher_student[:, -1:] = self.erosion(teacher_student[:, -1:])
            teacher_student[:, -1:] = self.dilation(teacher_student[:, -1:])
            teacher_student[:, -1:] = self.dilation(teacher_student[:, -1:])

            teacher_student[:, :-1] = self.erosion_3(teacher_student[:, :-1])
            teacher_student[:, :-1] = self.dilation_3(teacher_student[:, :-1])
            teacher_student[:, :-1] = self.dilation_3(teacher_student[:, :-1])
            #
            teacher_student = self.patchify(teacher_student)
        return teacher_student.mean(2)

    # ------------------------------------------------------------------
    # Clip-level Temporal Joint BW² (separate from forward_loss_TS above)
    # ------------------------------------------------------------------

    def uses_temporal_joint_ts(self):
        return self.ts_loss_type in (
            "temporal_joint",
            "temporal_joint_mse",
            "temporal_joint_bw2mse",
            "temporal_joint_bw2lr_mse",
        )

    def _encode_decode_ts_frame(self, img, grad_mask, mask_ratio):
        latent, mask, ids_restore = self.forward_encoder(img, mask_ratio, grad_mask)
        pred_stud, pred_teacher = self.forward_decoder_TS(latent, ids_restore)
        return pred_stud, pred_teacher, mask

    def forward_loss_temporal_joint(self, preds_stud, preds_teacher, masks):
        from util.temporal_joint_bw2 import temporal_joint_distill_loss

        batch_size, clip_len = preds_stud.shape[:2]
        flat_stud = preds_stud.reshape(batch_size * clip_len, preds_stud.shape[2], preds_stud.shape[3])
        flat_teacher = preds_teacher.reshape(
            batch_size * clip_len, preds_teacher.shape[2], preds_teacher.shape[3]
        )
        flat_masks = masks.reshape(batch_size * clip_len, masks.shape[2])
        per_frame_mse = self._ts_patch_mse(flat_stud, flat_teacher, flat_masks)
        if self.ts_loss_type in ("temporal_joint_bw2mse", "temporal_joint_bw2lr_mse"):
            per_sample_bw2, _ = self._ts_distill_per_sample(
                flat_stud, flat_teacher, flat_masks
            )
            per_frame_loss = self._combine_bw2_mse(per_frame_mse, per_sample_bw2)
        else:
            per_frame_loss = per_frame_mse
        per_frame_loss = per_frame_loss.view(batch_size, clip_len)

        return temporal_joint_distill_loss(
            preds_stud,
            preds_teacher,
            masks,
            per_frame_loss,
            loss_type=self.ts_loss_type,
            joint_lambda=self.ts_joint_lambda,
            joint_rank=self.ts_joint_rank,
            joint_stat=self.ts_joint_stat,
            eps=self.bw2_eps,
        )

    def forward_clip_TS(
        self,
        imgs,
        targets,
        grad_mask=None,
        mask_ratio=0.75,
        is_abnormal=None,
    ):
        """
        Stage-2 clip forward for Temporal Joint BW².

        imgs: [B, K, C, H, W]
        Returns the same tuple as training forward: (loss, pred_stud, mask)
        """
        if not self.uses_temporal_joint_ts():
            raise ValueError(
                f"forward_clip_TS requires ts_loss_type temporal_joint*, got {self.ts_loss_type}"
            )
        if imgs.dim() != 5:
            raise ValueError(f"forward_clip_TS expects imgs [B,K,C,H,W], got shape {tuple(imgs.shape)}")

        batch_size, clip_len = imgs.shape[:2]
        preds_stud, preds_teacher, masks = [], [], []
        last_pred_stud, last_mask = None, None

        for k in range(clip_len):
            pred_stud, pred_teacher, mask = self._encode_decode_ts_frame(
                imgs[:, k], grad_mask[:, k], mask_ratio
            )
            preds_stud.append(pred_stud)
            preds_teacher.append(pred_teacher)
            masks.append(mask)
            last_pred_stud, last_mask = pred_stud, mask

        preds_stud = torch.stack(preds_stud, dim=1)
        preds_teacher = torch.stack(preds_teacher, dim=1)
        masks = torch.stack(masks, dim=1)

        loss = self.forward_loss_temporal_joint(preds_stud, preds_teacher, masks)

        if self.training:
            return loss, last_pred_stud, last_mask

        last_k = clip_len - 1
        return (
            loss,
            preds_teacher[:, last_k],
            masks[:, last_k],
            self.abnormal_score_TS(
                targets[:, last_k],
                preds_stud[:, last_k],
                preds_teacher[:, last_k],
                masks[:, last_k],
                grad_mask[:, last_k],
            ),
        )
