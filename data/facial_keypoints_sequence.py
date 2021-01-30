import math

from keras.utils.data_utils import Sequence
import numpy as np


class FacialKeypointsSequence(Sequence):
    def __init__(self, fc_dataset, batch_size):
        self.fc_dataset = fc_dataset
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.fc_dataset) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for i in range(self.batch_size):
            x, y = self.fc_dataset[idx*self.batch_size + i]
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)
