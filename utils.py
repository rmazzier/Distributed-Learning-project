"""Utility Functions"""

import enum
import torch
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from constants import DEVICE


class SPLIT(enum.Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def backtracking_line_search(X, y, criterion, parameters, gradient, direction, c=0.1):
    candidates = [10000, 1000, 100, 10, 1.0, 0.1, 0.01]
    for candidate in candidates:
        # Armijo - Goldstein condition
        y_hat_old = torch.matmul(
            X, torch.transpose(parameters, 0, 1)).squeeze()
        y_hat_new = torch.matmul(
            X, torch.transpose(parameters + candidate * direction, 0, 1)
        ).squeeze()

        if criterion(y_hat_new, y, parameters + candidate * direction) <= criterion(
            y_hat_old, y, parameters
        ) + candidate * c * torch.dot(direction.squeeze(), gradient.squeeze()):
            break
    # print(f"Step size: {candidate}")
    return candidate


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return torch.tensor(y1).float().to(DEVICE)

        y = y1
