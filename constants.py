"""Configuration variables and hyperparameters"""

import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # Paths
    "RAW_DATA_PATH_EPSILON": os.path.join("data", "epsilon", "raw", "data"),
    "RAW_TEST_DATA_PATH_EPSILON": os.path.join("data", "epsilon", "raw", "data_test"),
    "EPSILON_SAMPLES_PATH": os.path.join("data", "epsilon", "samples"),
    "GEN_DATA_DIR": os.path.join("data", "samples"),
    "RESULTS_DIR": os.path.join("results"),
    # Dataset parameters
    "SAMPLES_PER_AGENT": 10000,
    "SPLIT_SIZES": [0.8, 0.1, 0.1],  # must sum to 1
    "EPOCHS": 20,
    "BATCH_SIZE": 128,
    # --- WANDB VARIABLES ---
    "MODEL_NAME": "S2S_Final_FedProx",
    "WANDB_MODE": "online",
    "WANDB_GROUP": "",
    "WANDB_TAGS": [],
    "NOTES": "",
}
