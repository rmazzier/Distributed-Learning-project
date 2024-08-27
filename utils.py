"""Utility Functions"""

import enum
import torch


class SPLIT(enum.Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def backtracking_line_search(X, y, criterion, parameters, gradient, direction, c=0.1):
    candidates = [10000, 1000, 100, 10, 1.0, 0.1, 0.01]
    for candidate in candidates:
        # Armijo - Goldstein condition
        y_hat_old = torch.matmul(X, torch.transpose(parameters, 0, 1)).squeeze()
        y_hat_new = torch.matmul(
            X, torch.transpose(parameters + candidate * direction, 0, 1)
        ).squeeze()

        if criterion(y_hat_new, y, parameters + candidate * direction) <= criterion(
            y_hat_old, y, parameters
        ) + candidate * c * torch.dot(direction.squeeze(), gradient.squeeze()):
            break
    # print(f"Step size: {candidate}")
    return candidate
