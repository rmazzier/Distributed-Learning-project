"""Configuration variables and hyperparameters"""

import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # Paths
    "EPSILON_DATASET_PATH": os.path.join("data", "epsilon"),
    "RAW_DATA_PATH_EPSILON": os.path.join("data", "epsilon", "raw", "data"),
    "RAW_TEST_DATA_PATH_EPSILON": os.path.join("data", "epsilon", "raw", "data_test"),
    "EPSILON_SAMPLES_PATH": os.path.join("data", "epsilon", "samples"),
    "GEN_DATA_DIR": os.path.join("data", "samples"),
    "RESULTS_DIR": os.path.join("results"),
    # Dataset parameters
    "N_CLIENTS": 3,
    "N_BYZANTINE_CLIENTS": 1,
    "MAX_CLIENT_ITERS": 50,
    # (Train, Validation), test is already separated
    "SPLIT_SIZES": [0.8, 0.2],
    "EPOCHS": 20,
    "NUM_ROUNDS": 20,
    "BATCH_SIZE": 128,
    "GAMMA": 1e-3,
    "USE_RANDOM_FOURIER": False,
    "N_FOURIER_FEATURES": 10000,
    # --- WANDB VARIABLES ---
    "MODEL_NAME": "ByzantineTest",
    "WANDB_MODE": "disabled",
    "WANDB_GROUP": "tests",
    "WANDB_TAGS": [],
    "NOTES": "First test",
}
