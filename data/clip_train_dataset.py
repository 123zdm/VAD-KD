"""K-frame clip training dataset (separate from single-frame AbnormalDatasetGradientsTrain)."""

from __future__ import annotations

import os
import random

import cv2
import numpy as np

from data.train_dataset import AbnormalDatasetGradientsTrain


class ClipDatasetGradientsTrain(AbnormalDatasetGradientsTrain):
    """
    Sample contiguous K-frame clips from the same training video.

    Does not modify the parent single-frame __getitem__; clip loading is
    implemented via _load_single_frame().
    """

    def __init__(self, args):
        super().__init__(args)
        self.clip_len = int(args.clip_len)
        self.clip_stride = int(getattr(args, "clip_stride", 1))
        if self.clip_len < 2:
            raise ValueError(f"clip_len must be >= 2 for ClipDataset, got {self.clip_len}")
        self.clips = self._build_clip_index()

    def _build_clip_index(self) -> list[list[int]]:
        video_to_indices: dict[str, list[int]] = {}
        for idx, path in enumerate(self.data):
            dir_path = "/".join(path.split("/")[:-1])
            video_to_indices.setdefault(dir_path, []).append(idx)

        clips: list[list[int]] = []
        for indices in video_to_indices.values():
            indices = sorted(
                indices,
                key=lambda i: int(self.data[i].split("/")[-1].split(".")[0]),
            )
            for start in range(0, len(indices) - self.clip_len + 1, self.clip_stride):
                clips.append(indices[start : start + self.clip_len])
        if not clips:
            raise RuntimeError(
                f"No clips of length {self.clip_len} could be built from training data."
            )
        return clips

    def _load_single_frame(self, index: int):
        """Load one frame with the same logic as the parent dataset."""
        random_uniform = random.uniform(0, 1)
        use_abnormal = (
            random_uniform <= self.percent_abnormal
            and self.abnormal_available
            and os.path.exists(self.abnormal_data[index])
            and os.path.exists(self.masks_abnormal[index])
        )
        if use_abnormal:
            img = self._read_required_image(self.abnormal_data[index], "abnormal frame")
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.abnormal_data, index)
            previous_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=-3, length=len_frame_no
            )
            next_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=3, length=len_frame_no
            )
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            mask = self._read_required_image(self.masks_abnormal[index], "abnormal mask")[:, :, :1]
        else:
            img = self._read_required_image(self.data[index], "normal frame")
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
            previous_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=-3, length=len_frame_no
            )
            next_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=3, length=len_frame_no
            )
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

        gradient = self._read_required_image(
            self.gradients[index],
            "gradient map (please run extract_gradients.py before training)",
        )
        target = self._read_required_image(self.data[index], "target frame")

        if img.shape[:2] != self.args.input_size or gradient.shape[:2] != self.args.input_size:
            img = cv2.resize(img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])
            mask = cv2.resize(mask, self.args.input_size[::-1])
            mask = np.expand_dims(mask, axis=-1)
        if target.shape[:2] != self.args.input_size:
            target = cv2.resize(target, self.args.input_size[::-1])

        target = np.concatenate((target, mask), axis=-1)
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        target = (target - 127.5) / 127.5
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        is_abnormal = np.float32(1.0 if use_abnormal else 0.0)
        return img, gradient, target, is_abnormal

    def __getitem__(self, clip_idx):
        frame_indices = self.clips[clip_idx]
        imgs, grads, targets, abnormals = [], [], [], []
        for frame_idx in frame_indices:
            img, grad, target, is_abnormal = self._load_single_frame(frame_idx)
            imgs.append(img)
            grads.append(grad)
            targets.append(target)
            abnormals.append(is_abnormal)

        return (
            np.stack(imgs, axis=0),
            np.stack(grads, axis=0),
            np.stack(targets, axis=0),
            np.stack(abnormals, axis=0),
        )

    def __len__(self):
        return len(self.clips)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(clip_len={self.clip_len}, "
            f"num_clips={len(self.clips)})"
        )
