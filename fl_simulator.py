"""
Classes and functions for a custom made federated learning simulator.
"""

import torch

# from torch.autograd.functional import hessian
import numpy as np
import os
import pathlib
from tqdm import tqdm
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt


from typing import List, Tuple
from collections import OrderedDict

from constants import CONFIG, DEVICE
from utils import SPLIT
from dataset import EpsilonDataset
from model import LinearModel
from train import (
    LRLoss_simple,
    train_epoch_SGD,
    BCELossWithLogits_simple,
    BCELossWithLogits_simple_dxx,
)


class Client:
    def __init__(self, config, client_idx):
        print(f"Initializing client {client_idx}")
        self.client_idx = client_idx
        self.config = config

        # Initialize local datasets
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
            input_dim,
        ) = self.initialize_local_datasets(config, client_idx)

        # Initialize the model
        self.net = LinearModel(input_dim)

        # Initialize Approximate newton direction member variable
        self.ant: torch.Tensor = None

        # Initialize local gradient member variable
        self.local_gradient: torch.Tensor = None

        # Initialize loss function
        self.criterion = BCELossWithLogits_simple
        # self.criterion = LRLoss_simple

    def initialize_local_datasets(self, config, client_idx):
        train_dataset = EpsilonDataset(config, client_idx, SPLIT.TRAIN)
        valid_dataset = EpsilonDataset(config, client_idx, SPLIT.VALIDATION)
        test_dataset = EpsilonDataset(config, client_idx, SPLIT.TEST)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            drop_last=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            drop_last=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            drop_last=True,
        )

        # Take single value to get input dimension
        x, _ = train_dataset[0]
        input_dim = x.shape[0]

        return train_loader, valid_loader, test_loader, input_dim

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set the parameters of the network.
        This is used to set the global parameters to a client.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the network.
        """

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def compute_local_gradient(self, is_byzantine=False) -> List[np.ndarray]:
        """
        Compute the gradient from the local dataset.
        """
        self.net.train()

        # Reset gradients of network, to be sure gradients are zero
        # NB: We are going to compute the gradient for the whole local dataset.
        # To do it efficiently, we accumulate the gradients
        # i.e. we do not need to zero the gradients at each iteration.
        self.net.zero_grad()

        counter = 0

        print("Start train dataloader for loop")
        for X, y in self.train_dataloader:
            if counter % 10 == 0:
                print(f"Client {self.client_idx}, Batch {counter}")
            X = X.float().to(DEVICE)
            y = y.float().to(DEVICE)

            # print("Forward pass...")
            y_pred = self.net(X).squeeze()

            # Loss computation
            l2_norm = torch.tensor(0.0).to(DEVICE)
            for param in self.net.parameters():
                l2_norm += torch.linalg.vector_norm(param, ord=2) ** 2

            loss = self.criterion(y_pred, y) + self.config["GAMMA"] / 2 * l2_norm

            loss.backward()

            counter += 1

        # Set the local gradient member variable to the accumulated gradient
        if is_byzantine:
            # return random stuff (check again)
            lg = [
                torch.randn(param.shape).to(DEVICE) for param in self.net.parameters()
            ]
            self.local_gradient = torch.cat(lg, dim=0).to(DEVICE)
        else:
            lg = [param.grad.cpu() for param in self.net.parameters()]
            self.local_gradient = torch.cat(lg, dim=0).to(DEVICE)

    def compute_ant(self, aggregated_gradient: torch.Tensor) -> List[np.ndarray]:

        # Compute the Hessian of the loss function
        self.net.train()
        self.net.zero_grad()

        ajs = []

        counter = 0

        for X, y in self.train_dataloader:
            if counter % 10 == 0:
                print(f"(ANT-Pass) Client {self.client_idx}, Batch {counter}")
            X = X.float().to(DEVICE)
            y = y.float().to(DEVICE)

            y_pred = self.net(X)

            dxx_l = BCELossWithLogits_simple_dxx(y_pred)

            ajs_batch = torch.sqrt(dxx_l) * X

            ajs.append(ajs_batch)
            counter += 1

        A_j = torch.cat(ajs, dim=0).to(DEVICE)
        s = len(self.train_dataloader) * self.config["BATCH_SIZE"]
        h_tilde = 1 / s * torch.matmul(A_j.transpose(0, 1), A_j) + self.config[
            "GAMMA"
        ] * torch.eye(A_j.shape[1]).to(DEVICE)

        # We now have to solve the following linear system:
        #        h_tilde * ant = aggregated_gradient
        # GIANT solves this using the Conjugate Gradient method (CG)
        ant, _ = cg(
            h_tilde.detach().cpu().numpy(),
            aggregated_gradient.cpu().numpy(),
            maxiter=1000,
        )

        # Set the ant member variable to the computed ant
        self.ant = torch.tensor(ant).squeeze().to(DEVICE)
        pass


class Server:
    def __init__(self, config) -> None:
        self.config = config
        self.clients = self.initialize_clients(config)
        pass

    def initialize_clients(self, config):
        return [Client(config, i) for i in range(config["N_CLIENTS"])]

    def aggregate_gradients(self):
        local_gradients = [client.local_gradient for client in self.clients]
        # return np.sum(local_gradients, axis=0)
        return np.mean(local_gradients, axis=0)

    def aggregate_ants(self):
        """
        Aggregate the approximate newton directions from the clients.
        i.e. Compute the GIANT (Globally Improved Approximate Newton Direction)
        """
        ants = [client.ant for client in self.clients]
        return np.mean(ants, axis=0)


if __name__ == "__main__":

    client_test = Client(CONFIG, 0)
    client_test.compute_local_gradient()
    local_grad = client_test.local_gradient
    params = client_test.get_parameters()

    client_test.compute_ant(local_grad)
