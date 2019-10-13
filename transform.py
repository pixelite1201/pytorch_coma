import numpy as np
import torch

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data


class NormalizeAmass(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(np.mean(data.x, axis=0), dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(np.std(data.x, axis=0), dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data

class NormalizeScan(object):
    def __init__(self, mean_x=None, std_x=None, mean_y=None, std_y=None):
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

    def set_mean_y(self, mean_y):
        self.mean_y = mean_y

    def set_std_y(self, std_y):
        self.std_y = std_y

    def __call__(self, data):
        assert self.mean_x is not None and self.std_x is not None, ('Initialize mean_x and std_x to normalize with')
        assert self.mean_y is not None and self.std_y is not None, ('Initialize mean_y and std_y to normalize with')

        self.mean_x = torch.as_tensor(np.mean(data.x, axis=0), dtype=data.x.dtype, device=data.x.device)
        self.std_x = torch.as_tensor(np.std(data.x, axis=0), dtype=data.x.dtype, device=data.x.device)
        self.mean_y = torch.as_tensor(self.mean_y, dtype=data.y.dtype, device=data.x.device)
        self.std_y = torch.as_tensor(self.std_y, dtype=data.y.dtype, device=data.x.device)

        data.x = (data.x - self.mean_x)/self.std_x
        data.y = (data.y - self.mean_y)/self.std_y
        return data
