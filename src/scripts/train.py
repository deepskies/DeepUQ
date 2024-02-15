import argparse
import torch
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from scripts import models


def train_DER(
    trainDataLoader,
    x_val,
    y_val,
    INIT_LR,
    DEVICE,
    COEFF,
    DER_type,
    model_name,
    EPOCHS=40,
    save_checkpoints=False,
    path_to_model="models/",
    plot=False,
):
    # measure how long training is going to take
    print("[INFO] training the network...")

    print("saving checkpoints?")
    print(save_checkpoints)

    startTime = time.time()

    # Find last epoch saved
    if save_checkpoints:

        print(glob.glob(path_to_model + "/" + str(model_name) + "*"))
        list_models_run = []
        for file in glob.glob(path_to_model + "/" + str(model_name) + "*"):
            list_models_run.append(
                float(str.split(str(str.split(file,
                                              model_name + "_")[1]), ".")[0])
            )
        if list_models_run:
            start_epoch = max(list_models_run) + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    print("starting here", start_epoch)

    loss_all_epochs = []  # this is from the training set
    loss_validation = []

    best_loss = np.inf  # init to infinity

    model, lossFn = models.model_setup_DER(DER_type, DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # loop over our epochs
    for e in range(0, EPOCHS):
        epoch = int(start_epoch + e)

        # set the model in training mode
        model.train()

        # loop over the training set
        print("epoch", epoch, round(e / EPOCHS, 2))

        loss_this_epoch = []

        plt.clf()
        # randomly shuffles the training data (if shuffle = True)
        # and draws batches up to the total training size
        # (should be about 8 batches)
        for i, (x, y) in enumerate(trainDataLoader):
            # print('i', i, len(y))
            # send the input to the device
            # (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss

            pred = model(x)
            loss = lossFn(pred, y, COEFF)
            if plot:
                if e % 5 == 0:
                    if i == 0:
                        # if loss_type == 'no_var_loss':
                        plt.scatter(
                            y,
                            pred[:, 0].flatten().detach().numpy(),
                            color="#F45866",
                            edgecolor="black",
                            zorder=100,
                        )
                        plt.errorbar(
                            y,
                            pred[:, 0].flatten().detach().numpy(),
                            yerr=loss[2],
                            color="#F45866",
                            zorder=100,
                            ls="None",
                        )
                        plt.annotate(
                            r"med $u_{ep} = " + str(np.median(loss[2])),
                            xy=(0.03, 0.93),
                            xycoords="axes fraction",
                            color="#F45866",
                        )

                    else:
                        plt.scatter(y, pred[:, 0].flatten().detach().numpy())
            loss_this_epoch.append(loss[0].item())

            # zero out the gradients
            opt.zero_grad()
            # perform the backpropagation step
            # computes the derivative of loss with respect to the parameters
            loss[0].backward()
            # update the weights
            # optimizer takes a step based on the gradients of the parameters
            # here, its taking a step for every batch
            opt.step()
        if plot:
            if e % 5 == 0:
                plt.ylabel("prediction")
                plt.xlabel("true value")
                plt.title("Epoch " + str(e))
                plt.show()
        loss_all_epochs.append(loss_this_epoch)
        # print('training loss', np.mean(loss_this_epoch))

        # this code from Rohan:
        # now, once an epoch is done:
        model.eval()
        # print('x val', x_val)
        # print('y val', y_val)
        y_pred = model(torch.Tensor(x_val))
        loss = lossFn(y_pred, torch.Tensor(y_val), COEFF)
        NIGloss_val = loss[0].item()
        med_u_al_val = np.median(loss[1])
        med_u_ep_val = np.median(loss[2])

        loss_validation.append(NIGloss_val)
        if NIGloss_val < best_loss:
            best_loss = NIGloss_val
            print("new best loss", NIGloss_val, "in epoch", epoch)
            # best_weights = copy.deepcopy(model.state_dict())
        # print('validation loss', mse)

        if save_checkpoints:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": np.mean(loss_this_epoch),
                    "valid_loss": NIGloss_val,
                    "med_u_al_validation": med_u_al_val,
                    "med_u_ep_validation": med_u_ep_val,
                },
                path_to_model + "/" + str(model_name) +
                "_epoch_" + str(epoch) + ".pt",
            )
    endTime = time.time()
    print("start at", startTime, "end at", endTime)
    print(endTime - startTime)

    return model


def train_DE(
    trainDataLoader,
    x_val,
    y_val,
    INIT_LR,
    DEVICE,
    loss_type,
    n_models,
    model_name="DE",
    EPOCHS=100,
    path_to_model="models/",
    save_checkpoints=False,
    plot=True,
    savefig=True,
):

    startTime = time.time()

    # Find last epoch saved
    if save_checkpoints is True:

        print('looking for saved checkpts',
              glob.glob("models/*" + model_name + "*"))
        list_models_run = []
        for file in glob.glob("models/*" + model_name + "*"):
            list_models_run.append(
                float(str.split(str(str.split(file,
                                              model_name + "_")[1]), ".")[0])
            )
        if list_models_run:
            start_epoch = max(list_models_run) + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    print("starting here", start_epoch)

    loss_all_epochs = []  # this is from the training set
    loss_validation = []

    best_loss = np.inf  # init to infinity

    model_ensemble = []

    for m in range(n_models):
        # initialize the model again each time from scratch
        model, lossFn, opt = models.model_setup_DE(loss_type, DEVICE, INIT_LR)

        # loop over our epochs
        for e in range(0, EPOCHS):
            epoch = int(start_epoch + e)

            # set the model in training mode
            model.train()

            # loop over the training set
            print("epoch", epoch, round(e / EPOCHS, 2))

            loss_this_epoch = []

            plt.clf()
            # randomly shuffles the training data (if shuffle = True)
            # and draws batches up to the total training size
            # (should be about 8 batches)
            for i, (x, y) in enumerate(trainDataLoader):
                # print('i', i, len(y))
                # send the input to the device
                # (x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss

                pred = model(x)
                if loss_type == "no_var_loss":
                    loss = lossFn(pred.flatten(), y)
                if loss_type == "var_loss":
                    loss = lossFn(pred[:, 0].flatten(),
                                  y,
                                  pred[:, 1].flatten() ** 2)
                if loss_type == "bnll_loss":
                    '''
                    if e/EPOCHS < 0.2:
                        # use beta = 1
                        beta_epoch = 1
                    if (e/EPOCHS > 0.2) & (e/EPOCHS < 0.5):
                        beta_epoch = 0.75
                    if e/EPOCHS > 0.5:
                        beta_epoch = 0.5
                    # 1 - e / EPOCHS # this one doesn't work great
                    '''
                    beta_epoch = 1 - e / EPOCHS
                    loss = lossFn(pred[:, 0].flatten(),
                                  pred[:, 1].flatten() ** 2,
                                  y,
                                  beta=beta_epoch)
                if plot is True:
                    if e % 5 == 0:
                        if i == 0:
                            plt.plot(range(0, 1000),
                                     range(0, 1000),
                                     color='black', ls='--')
                            if loss_type == "no_var_loss":
                                plt.scatter(
                                    y,
                                    pred.flatten().detach().numpy(),
                                    color="#F45866",
                                    edgecolor="black",
                                    zorder=100,
                                )
                            else:
                                plt.errorbar(
                                    y,
                                    pred[:, 0].flatten().detach().numpy(),
                                    yerr=abs(pred[:, 1].
                                             flatten().detach().numpy()),
                                    linestyle="None",
                                    color="black",
                                    capsize=2,
                                    zorder=100,
                                )
                                plt.scatter(
                                    y,
                                    pred[:, 0].flatten().detach().numpy(),
                                    color="black",
                                    zorder=100,
                                )
                            if loss_type == "bnll_loss":
                                plt.annotate(r'$\beta = $' +
                                             str(round(beta_epoch, 2)),
                                             xy=(0.03, 0.9),
                                             xycoords='axes fraction')
                        else:
                            if loss_type == "no_var_loss":
                                plt.scatter(y, pred.flatten().detach().numpy(),
                                            color='grey',
                                            alpha=0.5)
                            else:
                                plt.scatter(y, pred[:, 0].flatten().
                                            detach().numpy(),
                                            color='grey',
                                            alpha=0.5)

                loss_this_epoch.append(loss.item())

                # zero out the gradients
                opt.zero_grad()
                # perform the backpropagation step
                # computes the derivative of loss with respect
                # to the parameters
                loss.backward()
                # update the weights
                # optimizer takes a step based on the gradients
                # of the parameters
                # here, its taking a step for every batch
                opt.step()
            if plot is True:
                if e % 5 == 0:
                    plt.ylabel("prediction")
                    plt.xlabel("true value")
                    plt.title("Epoch " + str(e))
                    plt.xlim([0, 1000])
                    plt.ylim([0, 1000])
                    if savefig is True:
                        plt.savefig("../images/animations/" +
                                    str(model_name) + "_nmodel_" +
                                    str(m) + "_epoch_" +
                                    str(epoch) + ".png")
                    plt.show()

            loss_all_epochs.append(loss_this_epoch)
            # print('training loss', np.mean(loss_this_epoch))

            # this code from Rohan:
            # now, once an epoch is done:
            model.eval()
            y_pred = model(torch.Tensor(x_val))
            # print(y_pred.flatten().size(), torch.Tensor(y_valid).size())
            if loss_type == "no_var_loss":
                loss = lossFn(y_pred.flatten(), torch.Tensor(y_val)).item()
            if loss_type == "no_var_loss":
                loss = lossFn(
                    y_pred[:, 0].flatten(),
                    torch.Tensor(y_val),
                    y_pred[:, 1].flatten() ** 2,
                ).item()
            if loss_type == "bnll_loss":
                loss = lossFn(
                    y_pred[:, 0].flatten(),
                    y_pred[:, 1].flatten() ** 2,
                    torch.Tensor(y_val),
                ).item()

            loss_validation.append(loss)
            if loss < best_loss:
                best_loss = loss
                print("new best loss", loss, "in epoch", epoch)
                # best_weights = copy.deepcopy(model.state_dict())
            # print('validation loss', mse)

            if save_checkpoints is True:

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": np.mean(loss_this_epoch),
                        "valid_loss": loss,
                        "valid_mean": y_pred[:, 0].flatten(),
                        "valid_sigma": y_pred[:, 1].flatten(),
                        "x_val": x_val,
                        "y_val": y_val,
                    },
                    path_to_model + "/" +
                    str(model_name) + "_nmodel_" +
                    str(m) + "_epoch_" +
                    str(epoch) + ".pt",
                )

        model_ensemble.append(model)

    endTime = time.time()
    print("start at", startTime, "end at", endTime)
    print(endTime - startTime)

    return model_ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str,
                        help="Data used to train the model")
    parser.add_argument(
        "--n_epochs", type=int,
        help="Integer number of epochs to train the model"
    )

    args = parser.parse_args()

    # eventually change the bottom to train_model,
    # which will contain train_DE and train_DER
    train_DER(data_source=args.data_source, n_epochs=args.n_epochs)
