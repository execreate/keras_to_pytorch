"""Dataset for facial keypoint detection"""

import os

import pandas as pd
import numpy as np

from .base_dataset import BaseDataset


class FacialKeypointsDataset(BaseDataset):
    """Dataset for facial keypoint detection"""
    def __init__(self, *args, train=True, transform=None, data_format="channels_last", **kwargs):
        super().__init__(*args, **kwargs)
        file_name = "training.csv" if train else "val.csv"
        csv_file = os.path.join(self.root_path, file_name)
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.data_format = data_format

    @staticmethod
    def _get_image(idx, key_pts_frame, data_format="channels_last"):
        img_str = key_pts_frame.loc[idx]['Image']
        img = np.array([
            int(item) for item in img_str.split()
        ]).reshape((96, 96))
        if data_format == "channels_last":
            return np.expand_dims(img, axis=2).astype(np.uint8) / 255.
        else:
            return np.expand_dims(img, axis=0).astype(np.uint8) / 255.

    @staticmethod
    def _get_keypoints(idx, key_pts_frame, shape=(15, 2)):
        keypoint_cols = list(key_pts_frame.columns)[:-1]
        key_pts = key_pts_frame.iloc[idx][keypoint_cols].values.reshape(shape)
        key_pts = (key_pts.astype(np.float) - 48.0) / 48.0
        return np.array(key_pts)

    def __len__(self):
        return self.key_pts_frame.shape[0]

    def __getitem__(self, idx):
        image = self._get_image(idx, self.key_pts_frame, data_format=self.data_format)
        keypoints = self._get_keypoints(idx, self.key_pts_frame)
        if self.transform:
            image = self.transform(image)
        return image, keypoints
