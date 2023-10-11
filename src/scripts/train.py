"""
Simple stubs to use for re-train of the final model
Can leave a default data source, or specify that 'load data' loads the dataset used in the final version
"""
import argparse
import torch
import sbi


def architecture():
    """
    :return: compiled architecture of the model you want to have trained
    """
    return 0

def load_data(data_source):
    """
    :return: data loader or full training data, split in val and train
    """
    return 0, 0

def train_model(data_source, n_epochs):
    """
    :param data_source:
    :param n_epochs:
    :return: trained model, or simply None, but saved trained model
    """
    data = load_data(data_source)
    model = architecture()

    return 0


def train_SBI_hierarchical(thetas, xs, prior):
    # Now let's put them in a tensor form that SBI can read.
    theta = torch.tensor(thetas, dtype=torch.float32)
    x = torch.tensor(xs, dtype=torch.float32)

    # instantiate the neural density estimator
    neural_posterior = sbi.utils.posterior_nn(model='maf')#,
                                  #embedding_net=embedding_net,
                                  #hidden_features=hidden_features,
                                  #num_transforms=num_transforms)
    # setup the inference procedure with the SNPE-C procedure
    inference = sbi.inference.SNPE(prior=prior,
                                   density_estimator=neural_posterior,
                                   device="cpu")

    # now that we have both the simulated images and
    # parameters defined properly, we can train the SBI.
    density_estimator = inference.append_simulations(theta, x).train()
    return inference.build_posterior(density_estimator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, help="Data used to train the model")
    parser.add_argument("--n_epochs", type=int, help='Integer number of epochs to train the model')

    args = parser.parse_args()

    train_model(data_source=args.data_source, n_epochs=args.n_epochs)
