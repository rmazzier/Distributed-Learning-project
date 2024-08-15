import torch
import numpy as np
import os
from tqdm import tqdm

from constants import CONFIG


class EpsilonDataset(torch.utils.data.Dataset):
    def __init__(self, config, client_idx):
        self.config = config
        self.client_idx = client_idx

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
            save_path = os.path.join("data", "epsilon", "samples", f"eps_{i}_{y}.obj")
            np.save(save_path, x)

        print("Reading raw test data file")
        with open(test_path, "r") as f:
            data = f.readlines()

        print("Saving test samples")
        for i in tqdm(range(len(data))):
            x, y = get_epsilon_sample(i, data)
            save_path = os.path.join(
                "data", "epsilon", "samples", f"eps_test_{i}_{y}.obj"
            )
            np.save(save_path, x)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.path, self.samples[idx]))
        return sample


if __name__ == "__main__":

    EpsilonDataset.generate_samples(CONFIG)
