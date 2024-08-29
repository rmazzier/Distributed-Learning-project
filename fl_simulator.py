"""
Classes and functions for a custom made federated learning simulator.
"""

import torch
import wandb
import json

# from torch.autograd.functional import hessian
import numpy as np
import os
import pathlib
from tqdm import tqdm
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt


from typing import List, Tuple, Dict
from collections import OrderedDict

from constants import CONFIG, DEVICE
from utils import SPLIT, geometric_median
from dataset import EpsilonDataset
from model import LinearModel
from train import (
    LRLoss_simple,
    train_epoch_SGD,
    BCELossWithLogits_L2,
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
        self.net = LinearModel(input_dim).to(DEVICE)

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

    def set_parameters(self, parameters: torch.Tensor) -> None:
        """
        Set the parameters of the network.
        This is used to set the global parameters to a client.
        """
        parameters = [parameters.detach().cpu().numpy()]
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> torch.Tensor:
        """
        Get the parameters of the network.
        """
        params = [val.cpu().numpy()
                  for _, val in self.net.state_dict().items()][0]
        return torch.tensor(params).to(DEVICE)

    def compute_local_gradient(self, is_byzantine=False, max_iters=None) -> None:
        """
        Compute the gradient from the local dataset.
        """
        self.net.train()

        # Reset gradients of network, to be sure gradients are zero
        # NB: We are going to compute the gradient for the whole local dataset.
        # To do it efficiently, we accumulate the gradients
        # i.e. we do not need to zero the gradients at each iteration.
        self.net.zero_grad()

        for i, (X, y) in enumerate(self.train_dataloader):
            if max_iters is not None and (i + 1) > max_iters:
                print("Max iters reached")
                break

            if (i + 1) % 10 == 0:
                print(f"Client {self.client_idx}, Batch {i + 1}")

            X = X.float().to(DEVICE)
            y = y.float().to(DEVICE)

            # print("Forward pass...")
            y_pred = self.net(X).squeeze()

            # Loss computation
            l2_norm = torch.tensor(0.0).to(DEVICE)
            for param in self.net.parameters():
                l2_norm += torch.linalg.vector_norm(param, ord=2) ** 2

            loss = self.criterion(y_pred, y) + \
                self.config["GAMMA"] / 2 * l2_norm

            loss.backward()
        # Set the local gradient member variable to the accumulated gradient
        if is_byzantine:
            print("eheh")
            # return random stuff (check again)
            fake_grad = [
                torch.randn(param.shape).to(DEVICE) for param in self.net.parameters()
            ]
            self.local_gradient = torch.cat(fake_grad, dim=0).to(DEVICE)
        else:
            lg = [param.grad.cpu() for param in self.net.parameters()]
            self.local_gradient = torch.cat(lg, dim=0).to(DEVICE)

    def compute_ant(
        self, aggregated_gradient: torch.Tensor, is_byzantine=False, max_iters=None
    ) -> None:

        # Compute the Hessian of the loss function
        self.net.eval()

        with torch.no_grad():

            ajs = []

            for i, (X, y) in enumerate(self.train_dataloader):
                if max_iters is not None and (i + 1) > max_iters:
                    print("Max iters reached")
                    break
                if i % 10 == 0:
                    print(f"(ANT-Pass) Client {self.client_idx}, Batch {i}")
                X = X.float().to(DEVICE)
                y = y.float().to(DEVICE)

                y_pred = self.net(X)

                dxx_l = BCELossWithLogits_simple_dxx(y_pred)

                ajs_batch = torch.sqrt(dxx_l) * X

                ajs.append(ajs_batch)
                i += 1

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
        if is_byzantine:
            # return random stuff (check again)
            fake_ant = [
                torch.randn(param.shape).to(DEVICE) for param in self.net.parameters()
            ]
            self.ant = torch.cat(fake_ant, dim=0).squeeze().to(DEVICE)
        else:
            self.ant = torch.tensor(ant).squeeze().to(DEVICE)

    def get_bls_losses(
        self,
        candidates: List[float],
        descent_direction: torch.Tensor,
        c=0.1,
        max_iters=None,
    ) -> torch.Tensor:
        """
        Compute losses for different step sizes.
        """
        bls_losses = []

        self.net.eval()
        with torch.no_grad():

            for candidate in candidates:
                print(f"Candidate {candidate}")
                l = 0
                for i, (X, y) in enumerate(self.train_dataloader):
                    if max_iters is not None and i >= max_iters:
                        break

                    X = X.float().to(DEVICE)
                    y = y.float().to(DEVICE)

                    # Get the current parameters as tensor
                    parameters = torch.cat(
                        [param.flatten() for param in self.net.parameters()]
                    )

                    new_candidate_params = parameters + candidate * descent_direction

                    y_hat_new = torch.matmul(
                        X, torch.transpose(
                            new_candidate_params.unsqueeze(0), 0, 1)
                    ).squeeze()

                    l += (
                        self.criterion(y_hat_new, y)
                        + self.config["GAMMA"]
                        * 0.5
                        * torch.linalg.vector_norm(new_candidate_params, ord=2) ** 2
                    )

                l = l / len(self.train_dataloader)
                bls_losses.append(l)
        return torch.tensor(bls_losses)

    def compute_local_loss(self, split: SPLIT) -> float:
        """
        Compute the loss from the local dataset.
        """
        self.net.eval()
        if split == SPLIT.TRAIN:
            dataloader = self.train_dataloader
        elif split == SPLIT.VALIDATION:
            dataloader = self.valid_dataloader
        elif split == SPLIT.TEST:
            dataloader = self.test_dataloader

        self.net.eval()

        l2_norm = 0
        for param in self.net.parameters():
            l2_norm += torch.linalg.vector_norm(param, ord=2) ** 2

        with torch.no_grad():
            loss = 0

            for X, y in dataloader:
                X = X.float().to(DEVICE)
                y = y.float().to(DEVICE)

                y_pred = self.net(X).squeeze()

                loss += self.criterion(y_pred, y) + \
                    self.config["GAMMA"] / 2 * l2_norm

            loss = loss / len(dataloader)
        return loss

    def compute_local_accuracy(self, split: SPLIT) -> float:
        """
        Compute the accuracy from the local dataset.
        """
        self.net.eval()
        if split == SPLIT.TRAIN:
            dataloader = self.train_dataloader
        elif split == SPLIT.VALIDATION:
            dataloader = self.valid_dataloader
        elif split == SPLIT.TEST:
            dataloader = self.test_dataloader

        self.net.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            for X, y in dataloader:
                X = X.float().to(DEVICE)
                y = y.float().to(DEVICE)

                y_pred = self.net(X).squeeze()

                correct += (y_pred > 0).eq(y > 0.5).sum().item()
                total += y.shape[0]

            accuracy = correct / total
        return accuracy


class Server:
    def __init__(
        self, config: Dict, reduce_op: str = "mean", start_parameters=None
    ) -> None:
        """
        Server class for the federated learning simulator.

        Parameters:
        - config: Configuration dictionary
        - reduce_op: Aggregation function to be used for the gradients. Can be
            either "mean" or "median".
        - start_parameters: Starting parameters for the global model. If None, the
            parameters are taken from the first client and set to all the clients.
        """
        self.config = config
        self.current_params = start_parameters

        if reduce_op in ["mean", "median"]:
            self.reduce_op = reduce_op
        else:
            raise ValueError(
                "Invalid reduce operation. Must be either 'mean' or 'median'."
            )

    def aggregate_gradients(self, local_gradients: torch.Tensor) -> torch.Tensor:
        if self.reduce_op == "mean":
            # Here we use GIANT's aggregation function
            return torch.mean(local_gradients, axis=0)
        elif self.reduce_op == "median":
            # Here we use MNM
            return geometric_median(local_gradients.cpu().numpy())
        else:
            raise ValueError("Invalid reduce operation.")

    def aggregate_ants(self, local_ants: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the approximate newton directions from the clients.
        i.e. Compute the GIANT (Globally Improved Approximate Newton Direction)
        """
        if self.reduce_op == "mean":
            return torch.mean(local_ants, axis=0)
        elif self.reduce_op == "median":
            return geometric_median(local_ants.cpu().numpy())
        else:
            raise ValueError("Invalid reduce operation.")

    def send_params_to_clients(self, clients: List[Client]) -> None:
        """
        Sync the global model parameters with the clients.
        """
        print("Sending parameters to clients")
        if self.current_params is None:
            self.current_params = clients[0].get_parameters()
        for client in clients:
            client.set_parameters(self.current_params)


# This class serves as an interface for the different strategies
class Strategy:
    def __init__(self) -> None:
        pass

    def fit_step(
        self, server: Server, clients: List[Client], byzantine_idxs: List[int]
    ) -> None:
        """
        A single fit step, where parameters of the global model are updated.
        """
        # To be implemented in the subclasses
        raise NotImplementedError

    def eval_step(self, server: Server, clients: List[Client]) -> None:
        """
        A single evaluation step, where the global model is evaluated.
        """
        # To be implemented in the subclasses
        raise NotImplementedError

    def backtracking_ls(
        self,
        clients: List[Client],
        gradient: torch.Tensor,
        descent_direction: torch.Tensor,
        max_iters: int = None,
    ) -> float:
        """
        Returns the optimal step size with Backtracking line search.
        """
        candidates = [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]

        candidate_losses = []
        local_losses = []
        for client in clients:
            candidate_losses.append(
                client.get_bls_losses(
                    candidates, descent_direction, c=0.1, max_iters=max_iters
                )
            )
            local_losses.append(client.compute_local_loss(SPLIT.TRAIN))

        candidate_losses = torch.stack(candidate_losses, dim=0).to(
            DEVICE
        )  # shape = (n_clients, n_candidates)

        local_losses = torch.tensor(local_losses)

        # Reduce operation, aka take the mean for each candidate
        mean_bls_losses = torch.mean(candidate_losses, axis=0)

        # Armijo-Goldstein condition
        # TODO: Check with Eleonora if this is correct
        for i, candidate_loss in enumerate(mean_bls_losses):
            if candidate_loss <= local_losses.mean() + candidates[i] * 0.1 * torch.dot(
                descent_direction.squeeze(), gradient.squeeze()
            ):
                break

        return candidates[i]


class GIANT(Strategy):
    def __init__(self, max_iters=None) -> None:
        super().__init__()
        self.max_iters = max_iters

    def fit_step(
        self, server: Server, clients: List[Client], byzantine_idxs: List[int]
    ) -> None:
        print("Starting GIANT fit step")
        # Single iteration of the GIANT algorithm

        # FIRST COMMUNICATION ROUND
        for client_idx, client in enumerate(clients):
            is_byzantine = client_idx in byzantine_idxs
            if is_byzantine:
                print(f"Client {client_idx} sending fake gradient")
            client.compute_local_gradient(
                max_iters=self.max_iters, is_byzantine=is_byzantine
            )

        gradient = server.aggregate_gradients(
            torch.stack([client.local_gradient.squeeze()
                        for client in clients], dim=0)
        )

        # SECOND COMMUNICATION ROUNDs
        for client_idx, client in enumerate(clients):
            is_byzantine = client_idx in byzantine_idxs
            if is_byzantine:
                print(f"Client {client_idx} sending fake ANT")
            client.compute_ant(
                gradient, max_iters=self.max_iters, is_byzantine=is_byzantine
            )

        newton_direction = server.aggregate_ants(
            torch.stack([client.ant.squeeze() for client in clients], dim=0)
        )

        # OPTIMIZATION STEP (i.e. final iteration of the Newton method)
        # NB: Requires two additional communication rounds
        step_size = self.backtracking_ls(
            clients, gradient, -newton_direction, max_iters=self.max_iters
        )
        print(f"Step size = {step_size}")
        server.current_params = server.current_params - step_size * newton_direction

        # Communicate the new parameters to the clients
        server.send_params_to_clients(clients)


class FedAvg(Strategy):
    def __init__(self, max_iters=None) -> None:
        super().__init__()
        self.max_iters = max_iters
        pass

    def fit_step(
        self, server: Server, clients: List[Client], byzantine_idxs: List[int]
    ) -> None:
        print("Starting FedAvg fit step")

        for client_idx, client in enumerate(clients):
            is_byzantine = client_idx in byzantine_idxs
            if is_byzantine:
                print(f"Client {client_idx} sending fake gradient")
            client.compute_local_gradient(
                max_iters=self.max_iters, is_byzantine=is_byzantine)

        gradient = server.aggregate_gradients(
            torch.stack([client.local_gradient.squeeze()
                        for client in clients], dim=0)
        )

        # OPTIMIZATION STEP (i.e. final iteration of the Newton method)
        # NB: Requires two additional communication rounds
        step_size = self.backtracking_ls(
            clients, gradient, -gradient, max_iters=self.max_iters
        )
        print(f"Step size = {step_size}")
        server.current_params = server.current_params - step_size * gradient

        # Communicate the new parameters to the clients
        server.send_params_to_clients(clients)


class Simulation:
    def __init__(
        self,
        config,
        strategy: Strategy,
        server: Server,
        n_clients: int,
        seed=0,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.rng = np.random.default_rng(seed=seed)

        self.clients = [Client(config, i) for i in range(n_clients)]

        # Plot parameters before and after syncing
        for j, c in enumerate(self.clients):
            params_numpy = [
                param.detach().cpu().numpy().squeeze() for param in c.net.parameters()
            ][0]
            plt.hist(params_numpy, bins=100)
            plt.title(f"0_Client {j} before syncing")
            plt.savefig(os.path.join(
                "figures", f"0_Client {j} before syncing.png"))
            plt.close()

        self.server = server

        # Equalize the parameters of all the clients
        self.server.send_params_to_clients(self.clients)

        for j, c in enumerate(self.clients):
            params_numpy = [
                param.detach().cpu().numpy().squeeze() for param in c.net.parameters()
            ][0]
            plt.hist(params_numpy, bins=100)
            plt.title(f"1_Client {j} after syncing")
            plt.savefig(os.path.join(
                "figures", f"1_Client {j} after syncing.png"))
            plt.close()

    def start_simulation(self, n_rounds: int) -> None:

        # Start wandb session
        wandb.login()
        run = wandb.init(
            project="DistributedOptimizationProject",
            config=self.config,
            name=self.config["MODEL_NAME"],
            notes=self.config["NOTES"],
            reinit=True,
            mode=self.config["WANDB_MODE"],
            group=self.config["WANDB_GROUP"],
            tags=self.config["WANDB_TAGS"],
        )

        train_loss, train_accuracy, valid_loss, valid_accuracy = (
            self.federated_evaluation()
        )

        wandb.log(
            {
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Valid Loss": valid_loss,
                "Valid Accuracy": valid_accuracy,
            },
            step=0,
        )

        print(
            f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}"
        )

        valid_losses = []

        for round in range(1, n_rounds + 1):

            # Select a random subset of corrupt byzantine clients
            # if self.config["N_BYZANTINE_CLIENTS"] > 0:
            byzantine_idxs = self.rng.choice(
                range(len(self.clients)),
                self.config["N_BYZANTINE_CLIENTS"],
                replace=False,
            )
            # else:
            #     byzantine_idxs = []

            print(f"Starting Round {round}")
            self.strategy.fit_step(self.server, self.clients, byzantine_idxs)

            # plot client parameters after syncing
            for j, c in enumerate(self.clients):
                params_numpy = [
                    param.detach().cpu().numpy().squeeze()
                    for param in c.net.parameters()
                ][0]
                plt.hist(params_numpy, bins=100)
                plt.title(f"2_Client {j} after syncing_{round}")
                plt.savefig(
                    os.path.join(
                        "figures", f"2_Client {j} after syncing_{round}.png")
                )
                plt.close()

            train_loss, train_accuracy, valid_loss, valid_accuracy = (
                self.federated_evaluation()
            )

            print(
                f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}"
            )

            # The round to be logged to depends on the strategy, since GIANT makes 2 steps,
            # while FedAvg makes only 1 step.
            # We ccheck if strategy is of type GIANT, if so we log the round number times 2
            if isinstance(self.strategy, GIANT):
                logged_round = round * 2
            else:
                logged_round = round

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                    "Valid Loss": valid_loss,
                    "Valid Accuracy": valid_accuracy,
                },
                step=logged_round,
            )

            # Check if this is the best validation loss, if so save the model
            valid_losses.append(valid_loss)
            if valid_loss == min(valid_losses):
                print("Saving best model")
                torch.save(
                    self.clients[0].net.state_dict(),
                    os.path.join(
                        "results",
                        f"best_model_weights.pt",
                    ),
                )

                # Also save the current config as a json file
                with open(os.path.join("results", "config.json"), "w") as config_file:
                    json.dump(self.config, config_file)

        # Final Test step here (i.e. take mean of accuracies and losses)
        print("Final Test step")
        test_loss = 0
        test_accuracy = 0

        for client in self.clients:
            test_loss += client.compute_local_loss(SPLIT.TEST)
            test_accuracy += client.compute_local_accuracy(SPLIT.TEST)

        test_loss = test_loss / len(self.clients)
        test_accuracy = test_accuracy / len(self.clients)

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        run.finish()

    def federated_evaluation(self):
        # Get training set loss and accuracy
        train_loss = 0
        train_accuracy = 0

        for client in self.clients:
            train_loss += client.compute_local_loss(SPLIT.TRAIN)
            train_accuracy += client.compute_local_accuracy(SPLIT.TRAIN)

        train_loss = train_loss / len(self.clients)
        train_accuracy = train_accuracy / len(self.clients)

        # Get validation set loss and accuracy
        valid_loss = 0
        valid_accuracy = 0

        for client in self.clients:
            valid_loss += client.compute_local_loss(SPLIT.VALIDATION)
            valid_accuracy += client.compute_local_accuracy(SPLIT.VALIDATION)

        valid_loss = valid_loss / len(self.clients)
        valid_accuracy = valid_accuracy / len(self.clients)
        return train_loss, train_accuracy, valid_loss, valid_accuracy


if __name__ == "__main__":

    # strategy = GIANT(CONFIG["MAX_CLIENT_ITERS"])
    strategy = FedAvg(CONFIG["MAX_CLIENT_ITERS"])

    server = Server(CONFIG, reduce_op="median")

    sim = Simulation(
        config=CONFIG, strategy=strategy, server=server, n_clients=CONFIG["N_CLIENTS"]
    )

    sim.start_simulation(n_rounds=CONFIG["NUM_ROUNDS"])
