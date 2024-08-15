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
    "N_CLIENTS": 20,
    "SPLIT_SIZES": [0.8, 0.2],  # (Train, Validation), test is already separated
    "EPOCHS": 20,
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 0.001,
    "GAMMA": 0.01,
    # --- WANDB VARIABLES ---
    "MODEL_NAME": "FirstTest",
    "WANDB_MODE": "disabled",
    "WANDB_GROUP": "",
    "WANDB_TAGS": [],
    "NOTES": "",
}
