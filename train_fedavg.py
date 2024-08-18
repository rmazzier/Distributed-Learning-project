import flwr as fl
import wandb
import os
import json

from dataset import EpsilonDataset

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
from constants import DEVICE, CONFIG
from fl_setup import (
    client_fn,
    server_fn,
    fit_config,
    FedAvgWandb,
    weighted_average_eval,
    weighted_average_fit,
)

# from utils import load_model_from_npz, plot_predictions_long
from train import setup_training


def train_federated(config, client, server):
    NUM_CLIENTS = config["N_CLIENTS"]
    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE == "cuda":
        # here we are assigning an entire GPU for each client.
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.

    # Initialize wandb configuration dict

    # PUT THIS RUN INSIDE THE "FedAvg" GROUP
    config["WANDB_GROUP"] = "FedAvg"
    config["MODEL_NAME"] = "FedAvg_" + config["MODEL_NAME"]

    # Save config.json file to folder
    run_results_dir = os.path.join(config["RESULTS_DIR"], config["MODEL_NAME"])

    # Check if it exists:
    # if os.path.exists(run_results_dir):
    #     print("Results directory already exists. Exiting...")
    #     return

    os.makedirs(run_results_dir, exist_ok=True)
    with open(os.path.join(run_results_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Start wandb session
    # wandb.login()
    # run = wandb.init(
    #     project="MLComSysProject",
    #     config=config,
    #     name=config["MODEL_NAME"],
    #     notes=config["NOTES"],
    #     reinit=True,
    #     mode=config["WANDB_MODE"],
    #     group=config["WANDB_GROUP"],
    #     tags=config["WANDB_TAGS"],
    # )

    # Start the actual simulation
    # fl.simulation.start_simulation(
    #     client_fn=lambda x: client_fn(x, config),
    #     num_clients=NUM_CLIENTS,
    #     config=fl.server.ServerConfig(num_rounds=n_rounds),
    #     strategy=strategy,
    #     client_resources=client_resources,
    # )

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=client_resources,
    )

    # Test dataset is the union of all the test sets of the clients
    # _, _, test_dataset, _, net = setup_training(config, agent_idx=-1)

    # net = load_model_from_npz(config, net)

    # plot_predictions_long(config, 100, 100, test_dataset, net)

    # run.finish()


if __name__ == "__main__":
    from constants import CONFIG

    EpsilonDataset.generate_agent_splits(CONFIG)

    # Create FedAvg strategy
    strategy = FedAvgWandb(
        my_config=CONFIG,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=4,  # Never sample less than 4 clients for training
        min_evaluate_clients=4,  # Never sample less than 4 clients for evaluation
        min_available_clients=4,  # Wait until all 4 clients are available
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
    )

    # Create the ClientApp and Serverapp
    client = ClientApp(client_fn=lambda x: client_fn(x, CONFIG))
    server = ServerApp(
        server_fn=lambda x: server_fn(x, strategy, rounds=CONFIG["NUM_ROUNDS"])
    )

    train_federated(CONFIG, client, server)
