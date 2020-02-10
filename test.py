from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import rotate_batch

import numpy as np
import os

# test_path = '~/data/yusun/cifar/CIFAR-10-C/frost.npy'
# print(os.path.exists(test_path))
correct = [1, 0, 1, 0, 1, 1]
correct = torch.cat(correct).numpy()
print(correct)