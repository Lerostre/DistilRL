import torch
import numpy as np

import pickle


def dump_pkl(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def generate_arm_dist(n_runs, n_arms, strategy="odd", mishap_chance=0.05):

    assert n_arms % 2 == 0
    mishap = np.random.randint(0, 100, (n_runs,)) <= mishap_chance * 100

    if strategy in ["odd", "even"]:
        probas = torch.zeros(n_runs, n_arms)
        probas[:, ::2] = 1
        probas[:, ::2] += torch.FloatTensor(n_runs, n_arms // 2).uniform_(
            -0.5, 0
        )
        probas[:, 1::2] += torch.FloatTensor(n_runs, n_arms // 2).uniform_(
            0, 0.5
        )
        if strategy == "odd":
            probas = 1 - probas
    elif strategy == "uniform":
        probas = torch.FloatTensor(n_runs, n_arms).uniform_(0, 1)
    probas[mishap] = torch.FloatTensor(mishap.sum(), n_arms).uniform_(0, 1)

    return probas.numpy()


def window_stack(array, window_size=(1, 3)):
    return np.vstack(
        np.lib.stride_tricks.sliding_window_view(array, window_size)
    )
