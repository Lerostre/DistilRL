import os
import random

import numpy as np
import torch

import sys


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def block_print():
    sys.stdout = open(os.devnull, "w")


def enable_print():
    sys.stdout = sys.__stdout__
