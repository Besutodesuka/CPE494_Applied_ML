import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
def scale(data, from_interval: Tuple[float, float], to_interval: Tuple[float, float] = (0, 1)):
    from_min, from_max = from_interval
    to_min, to_max = to_interval

    # small epsilon to prevent divide-by-zero
    eps = 1e-12  

    denom = (from_max - from_min)
    if abs(denom) < eps:
        denom = eps  # avoid inf/nan if all values are the same

    scaled_data = to_min + (data - from_min) * (to_max - to_min) / denom

    # replace any accidental nan or inf with safe numbers
    scaled_data = np.nan_to_num(scaled_data, nan=to_min, posinf=to_max, neginf=to_min)

    return scaled_data