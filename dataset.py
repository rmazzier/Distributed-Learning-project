import torch
import numpy as np
import os
import pathlib
from tqdm import tqdm

from constants import CONFIG
from utils import SPLIT


class EpsilonDataset(torch.utils.data.Dataset):
    def __init__(self, config, client_idx, split):
        self.config = config
        self.client_idx = client_idx

        self.filenames_filepath = os.path.join(
            config["EPSILON_DATASET_PATH"], f"agent_{client_idx}", f"{split.value}.txt"
        )

        self.filenames = open(self.filenames_filepath, "r").readlines()

        self.filepaths = [
            os.path.join(config["EPSILON_SAMPLES_PATH"], f.strip())
            for f in self.filenames
        ]

    @staticmethod
    def generate_samples(config):
        """
        Generate samples from the raw data files and save them as individual files
        """

        # Delete existing samples
        samples_path = config["EPSILON_SAMPLES_PATH"]
        if os.path.exists(samples_path):
            print("Deleting existing samples")
            for file in os.listdir(samples_path):
                os.remove(os.path.join(samples_path, file))
        else:
            os.makedirs(samples_path)

        def get_epsilon_sample(index, data):
            sample = data[index].strip().split()
            y = int(sample[0])
            x = [float(f.split(":")[1]) for f in sample[1:]]
            return np.array(x), y

        train_path = config["RAW_DATA_PATH_EPSILON"]
        test_path = config["RAW_TEST_DATA_PATH_EPSILON"]

        print("Reading raw training data file")
        with open(train_path, "r") as f:
            data = f.readlines()

        print("Saving training samples")
        for i in tqdm(range(len(data))):
            x, y = get_epsilon_sample(i, data)
            save_path = os.path.join("data", "epsilon", "samples", f"eps_{i}_{y}")
            np.save(save_path, x)

        print("Reading raw test data file")
        with open(test_path, "r") as f:
            data = f.readlines()

        print("Saving test samples")
        for i in tqdm(range(len(data))):
            x, y = get_epsilon_sample(i, data)
            save_path = os.path.join("data", "epsilon", "samples", f"eps_test_{i}_{y}")
            np.save(save_path, x)

    @staticmethod
    def generate_agent_splits(config, seed=0):
        rng = np.random.default_rng(seed)
        # Check if samples have been generated
        samples_dir = config["EPSILON_SAMPLES_PATH"]
        if len(os.listdir(samples_dir)) == 0:
            print("Samples are not generated. Run the generate_samples method first")
            return

        # Remove existing folders containing the "agent" subword and all its contents
        for folder in os.listdir(config["EPSILON_DATASET_PATH"]):
            if "agent" in folder:
                folder_path = os.path.join(config["EPSILON_DATASET_PATH"], folder)
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)

        all_samples = os.listdir(samples_dir)

        train_samples = [s for s in all_samples if "test" not in s]
        test_samples = [s for s in all_samples if "test" in s]

        samples_per_agent_train = int(
            (len(train_samples) // config["N_CLIENTS"]) * config["SPLIT_SIZES"][0]
        )
        samples_per_agent_valid = (
            len(train_samples) // config["N_CLIENTS"]
        ) - samples_per_agent_train
        samples_per_agent_test = len(test_samples) // config["N_CLIENTS"]

        print(f"N Agents: {config['N_CLIENTS']}")
        print(
            f"Samples per agent: {samples_per_agent_train + samples_per_agent_valid + samples_per_agent_test}"
        )
        print(f"Train samples per agent: {samples_per_agent_train}")
        print(f"Valid samples per agent: {samples_per_agent_valid}")
        print(f"Test samples per agent: {samples_per_agent_test}")

        for agent_idx in range(config["N_CLIENTS"]):
            # Create agent directory
            agent_dir = os.path.join(
                config["EPSILON_DATASET_PATH"], f"agent_{agent_idx}"
            )
            os.makedirs(agent_dir, exist_ok=True)

            # Randomly select train samples from all_samples
            agent_train_samples = rng.choice(
                train_samples, samples_per_agent_train, replace=False
            )
            train_samples = list(set(train_samples) - set(agent_train_samples))

            # Randomly select valid samples from remaining all_samples
            agent_valid_samples = rng.choice(
                train_samples, samples_per_agent_valid, replace=False
            )
            train_samples = list(set(train_samples) - set(agent_valid_samples))

            # Now select test samples
            agent_test_samples = rng.choice(
                test_samples, samples_per_agent_test, replace=False
            )
            test_samples = list(set(test_samples) - set(agent_test_samples))

            # Save the filenames in the agent directory as txt files
            print(f"Saving agent {agent_idx} samples")
            with open(os.path.join(agent_dir, "train.txt"), "w") as f:
                for sample in agent_train_samples:
                    f.write(sample + "\n")

            with open(os.path.join(agent_dir, "valid.txt"), "w") as f:
                for sample in agent_valid_samples:
                    f.write(sample + "\n")

            with open(os.path.join(agent_dir, "test.txt"), "w") as f:
                for sample in agent_test_samples:
                    f.write(sample + "\n")

        # Print leftover samples in train and test
        print(f"Leftover train samples: {len(train_samples)}")
        print(f"Leftover test samples: {len(test_samples)}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        sample = self.filepaths[idx]
        # Take just the file name (eps_test_88716_-1.obj.npy)
        fname = pathlib.Path(sample).name

        # Extract the label from the filename
        label = max(0, int(fname.split("_")[-1].split(".")[0]))
        # label = int(fname.split("_")[-1].split(".")[0])

        # Load the sample
        features = np.load(sample)

        return (features, label)


if __name__ == "__main__":

    dataset_train = EpsilonDataset(config=CONFIG, client_idx=0, split=SPLIT.TRAIN)
    dataset_valid = EpsilonDataset(config=CONFIG, client_idx=0, split=SPLIT.VALIDATION)
    dataset_test = EpsilonDataset(config=CONFIG, client_idx=0, split=SPLIT.TEST)

    print(len(dataset_train))
    print(len(dataset_valid))
    print(len(dataset_test))

    # Check that the filepaths of the three datasets don't overlap
    print(set(dataset_train.filepaths).intersection(dataset_valid.filepaths))
    print(set(dataset_train.filepaths).intersection(dataset_test.filepaths))
    print(set(dataset_valid.filepaths).intersection(dataset_test.filepaths))

    for x, y in dataset_train:
        print(x.shape, y)
        break
