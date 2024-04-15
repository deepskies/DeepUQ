# Contains modules used to prepare a dataset
# with varying noise properties
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from torch.distributions import Uniform
from torch.utils.data import TensorDataset
import torch
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="data handling module")
    parser.add_argument(
        "--arg",
        type=float,
        required=False,
        default=100,
        help="Description",
    )
    return parser.parse_args()


class ModelLoader:
    def save_model_pkl(self, path, model_name, posterior):
        """
        Save the pkl'ed saved posterior model

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + model_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(posterior, file)

    def load_model_pkl(self, path, model_name):
        """
        Load the pkl'ed saved posterior model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + model_name + ".pkl", "rb") as file:
            posterior = pickle.load(file)
        return posterior

    def predict(input, model):
        """

        :param input: loaded object used for inference
        :param model: loaded model
        :return: Prediction
        """
        return 0


# Example usage:
if __name__ == "__main__":
    namespace = parse_args()
    arg = namespace.arg
