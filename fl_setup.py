from collections import OrderedDict
from typing import List, Tuple
import os

# import matplotlib.pyplot as plt
import flwr as fl
from flwr.common import Metrics
import numpy as np
import torch
import torch.nn as nn
import wandb

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from typing import List, Optional, Tuple, Callable, Dict, Union
from flwr.common import (
    FitRes,
    FitIns,
    Context,
    MetricsAggregationFn,
    EvaluateRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


from constants import DEVICE, CONFIG
from train import train_epoch_SGD, test_step, setup_training

"""
We need two helper functions to update the local model with parameters received from
the server and to get the updated model parameters from the local model:
set_parameters and get_parameters.
The following two functions do that.
"""


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Set the parameters of the network.
    This is used to set the global parameters to a client.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    """
    Get the parameters of the network.
    Used when communicating from the client to the server.
    """

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


"""
In Flower, clients are subclasses of flwr.client.Client or flwr.client.NumPyClient.
(I still don't know the difference between the two)

Then, we must implement the following three methods:
- get_parameters: Get the current model parameters.
- fit: Update the model using the provided parameters and return the updated parameters to the server
- evaluate: Evaluate the model using the provided parameters and return evaluation to the server.
"""


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id,
        my_config,
        net,
        trainloader,
        valloader,
        criterion,
        max_iters=100,
    ):
        self.my_config = my_config
        self.client_id = client_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.criterion = criterion
        self.max_iters = max_iters

    def get_parameters(self, config):
        print(f"Getting Client {self.client_id} parameters!")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"Fitting Client {self.client_id}")
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.client_id}, round {server_round}] fit, config: {config}")

        set_parameters(self.net, parameters)
        train_loss, train_accuracy = train_epoch_SGD(
            self.my_config,
            self.trainloader,
            net=self.net,
            criterion=self.criterion,
            max_iters=100,
        )
        # This method returns:
        # 1: The updated parameters of the local client
        # 2: The size of the local client's dataset
        # 3: Training metrics, in this case loss and accuracy
        return (
            self.get_parameters({}),
            len(self.trainloader),
            {"train_loss": float(train_loss), "train_accuracy": float(train_accuracy)},
        )

    def evaluate(self, parameters, config):
        print(f"Evaluating Client {self.client_id}")
        set_parameters(self.net, parameters)
        valid_loss, valid_accuracy = test_step(self.valloader, self.net, self.criterion)
        return (
            float(valid_loss),
            len(self.valloader),
            {"valid_loss": float(valid_loss), "valid_accuracy": float(valid_accuracy)},
        )


def client_fn(context: Context, config) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    cid = context.node_config["partition-id"]

    print(f"Creating client {cid}")

    cid = int(cid)
    train_loader, valid_loader, _, _, net = setup_training(config, client_idx=cid)

    # Load model
    net = net.to(DEVICE)

    # optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Loss function
    if config["CRITERION"] == "MSE":
        criterion = nn.MSELoss()
    elif config["CRITERION"] == "MAE":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Invalid criterion")

    # Create a  single Flower client representing a single organization
    return FlowerClient(
        cid, config, net, train_loader, valid_loader, criterion, max_iters=50
    ).to_client()


def server_fn(context: Context, strategy, rounds) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["valid_loss"] for num_examples, m in metrics]
    accs = [num_examples * m["valid_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "valid_loss": sum(losses) / sum(examples),
        "valid_accuracy": sum(accs) / sum(examples),
    }


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    accs = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(losses) / sum(examples),
        "train_accuracy": sum(accs) / sum(examples),
    }


"""
Now I subclass flwr.server.strategy.FedAvg to be able to log things to wandb"""


class FedAvgWandb(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        my_config,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
        accept_failures: bool = True,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.my_config = my_config

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # wandb.log(metrics_aggregated, step=server_round)
        print(metrics_aggregated)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            save_dir = os.path.join(
                self.my_config["RESULTS_DIR"], self.my_config["MODEL_NAME"]
            )
            os.makedirs(save_dir, exist_ok=True)
            np.savez(
                os.path.join(save_dir, f"round-{server_round}-weights.npz"),
                *aggregated_ndarrays,
            )

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        print("Aggregating metrics")
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        wandb.log(metrics_aggregated, step=server_round)
        print(loss_aggregated, metrics_aggregated)
        return loss_aggregated, metrics_aggregated
