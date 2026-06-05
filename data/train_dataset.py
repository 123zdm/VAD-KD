import glob
import os
import random

import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]


class AbnormalDatasetGradientsTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        self.percent_abnormal = args.percent_abnormal
        self.input_3d = args.input_3d
        self.abnormal_available = os.path.isdir(os.path.join(data_path, "train", "frames_abnormal")) and \
                                  os.path.isdir(os.path.join(data_path, "train", "masks_abnormal"))
        if self.percent_abnormal > 0 and not self.abnormal_available:
            print(
                "Warning: frames_abnormal/masks_abnormal not found. "
                "Training will continue without synthetic anomalies."
            )
        self.abnormal_data, self.data, self.gradients, self.masks_abnormal = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        gradients = []
        abnormal_data = []
        masks_abnormal = []
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "train/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        if extension is None:
            raise FileNotFoundError(f"No training frames found in {os.path.join(data_path, 'train/frames')}")
        self.extension = extension

        dirs = sorted(glob.glob(os.path.join(data_path, "train", "frames", "*")))
        for dir in dirs:
            imgs_path = sorted(
                glob.glob(os.path.join(dir, f"*{extension}")),
                key=lambda x: int(os.path.basename(x).split('.')[0]),
            )
            data += imgs_path
            video_name = os.path.basename(dir)
            gradients_path = []
            for img_path in imgs_path:
                frame_name = os.path.basename(img_path)
                gradients_path.append(
                    os.path.join(data_path, "train", "gradients2", video_name, frame_name)
                )
                abnormal_data.append(
                    os.path.join(data_path, "train", "frames_abnormal", video_name, frame_name)
                )
                masks_abnormal.append(
                    os.path.join(data_path, "train", "masks_abnormal", video_name, frame_name)
                )
            gradients += gradients_path
        return abnormal_data, data, gradients, masks_abnormal

    @staticmethod
    def _read_required_image(path, image_name):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read {image_name}: {path}")
        return image

    def __getitem__(self, index):
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
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            mask = self._read_required_image(self.masks_abnormal[index], "abnormal mask")[:, :, :1]
        else:
            img = self._read_required_image(self.data[index], "normal frame")
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
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

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + self.extension
        if os.path.exists(frame_path):
            return self._read_required_image(frame_path, "neighbor frame")
        fallback_path = dir_path + "/" + str(frame_no).zfill(length) + self.extension
        return self._read_required_image(fallback_path, "fallback current frame")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__
