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
    verbose=True,
):
    # measure how long training is going to take
    if verbose:
        print("[INFO] training the network...")
        print("saving checkpoints?")
        print(save_checkpoints)
        print(f"saving here: {path_to_model}")
        print(f"model name: {model_name}")

    startTime = time.time()
    start_epoch = 0
    '''
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
    '''
    best_loss = np.inf  # init to infinity
    model, lossFn = models.model_setup_DER(DER_type, DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # loop over our epochs
    for e in range(0, EPOCHS):
        if plot:
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                            gridspec_kw={'height_ratios': [3, 1]}
                                            )
                
        epoch = int(start_epoch + e)

        # set the model in training mode
        model.train()

        # loop over the training set
        if verbose:
            print("epoch", epoch, round(e / EPOCHS, 2))
        loss_this_epoch = []
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
            if plot and (e % 5 == 0):
                if i == 0:
                    pred_loader_0 = pred[:,0].flatten().detach().numpy()
                    y_loader_0 = y.detach().numpy()
                    ax1.scatter(
                        y,
                        pred[:, 0].flatten().detach().numpy(),
                        color="black",
                        zorder=100,
                    )
                    ax1.errorbar(
                        y,
                        pred[:, 0].flatten().detach().numpy(),
                        yerr=loss[2],
                        color="black",
                        zorder=100,
                        ls="None",
                    )
                    ax1.annotate(
                        r"med $u_{ep} = $" + str(np.median(loss[2])),
                        xy=(0.03, 0.93),
                        xycoords="axes fraction",
                        color="black",
                    )
                else:
                    ax1.scatter(y,
                                pred[:, 0].flatten().detach().numpy(),
                                color='grey')
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
        if plot and (e % 5 == 0):
            ax1.set_ylabel("prediction")
            ax1.set_title("Epoch " + str(e))

            # Residuals plot
            residuals = pred_loader_0 - y_loader_0
            ax2.scatter(y_loader_0, residuals, color='red')
            ax2.axhline(0, color='black', linestyle='--', linewidth=1)
            ax2.set_ylabel("Residuals")
            ax2.set_xlabel("True Value")

            plt.show()
            plt.close()
        model.eval()
        y_pred = model(torch.Tensor(x_val))
        loss = lossFn(y_pred, torch.Tensor(y_val), COEFF)
        NIGloss_val = loss[0].item()
        med_u_al_val = np.median(loss[1])
        med_u_ep_val = np.median(loss[2])
        std_u_al_val = np.std(loss[1])
        std_u_ep_val = np.std(loss[2])

        # lets also grab mse loss
        mse_loss = torch.nn.MSELoss(reduction='mean')
        mse = mse_loss(y_pred[:,0], torch.Tensor(y_val)).item()
        if NIGloss_val < best_loss:
            best_loss = NIGloss_val
            if verbose:
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
                    "valid_mse": mse,
                    "med_u_al_validation": med_u_al_val,
                    "med_u_ep_validation": med_u_ep_val,
                    "std_u_al_validation": std_u_al_val,
                    "std_u_ep_validation": std_u_ep_val,
                },
                path_to_model + "/" + str(model_name) +
                "_epoch_" + str(epoch) + ".pt",
            )
    endTime = time.time()
    if verbose:
        print("start at", startTime, "end at", endTime)
        print(endTime - startTime)
    return model, mse


def train_DE(
    trainDataLoader,
    x_val,
    y_val,
    INIT_LR,
    DEVICE,
    loss_type,
    n_models,
    model_name="DE",
    BETA=None,
    EPOCHS=100,
    path_to_model="models/",
    save_all_checkpoints=False,
    save_final_checkpoint=False,
    plot=True,
    savefig=True,
    verbose=True,
):

    startTime = time.time()

    '''
    # Find last epoch saved
    if save_checkpoints:

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
    '''
    start_epoch = 0
    if verbose:
        print("starting here", start_epoch)

    loss_all_epochs = []  # this is from the training set
    loss_validation = []
    final_mse = []

    best_loss = np.inf  # init to infinity

    model_ensemble = []

    for m in range(n_models):
        print('model', m)
        # initialize the model again each time from scratch
        model, lossFn = models.model_setup_DE(loss_type, DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)

        # loop over our epochs
        for e in range(0, EPOCHS):
            plt.close()
            epoch = int(start_epoch + e)

            # set the model in training mode
            model.train()

            # loop over the training set
            if verbose:
                print("epoch", epoch, round(e / EPOCHS, 2))

            loss_this_epoch = []
            if plot:
                plt.clf()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                            gridspec_kw={'height_ratios': [3, 1]}
                                            )
             
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
                                  pred[:, 1].flatten())
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
                    #beta_epoch = 1 - e / EPOCHS
                    beta_epoch = BETA
                    loss = lossFn(pred[:, 0].flatten(),
                                  pred[:, 1].flatten(),
                                  y,
                                  beta=beta_epoch)
                if plot:
                    if (e % (EPOCHS-1) == 0) and (e != 0):
                        if loss_type == "no_var_loss":
                            ax1.scatter(y, pred.flatten().detach().numpy(),
                                        color='grey',
                                        alpha=0.5,
                                        label='training data')
                        else:
                            if i == 0:
                                ax1.scatter(y, pred[:, 0].flatten().
                                            detach().numpy(),
                                            color='grey',
                                            alpha=0.5,
                                            label='training data')
                            else:
                                ax1.scatter(y, pred[:, 0].flatten().
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
            

            loss_all_epochs.append(loss_this_epoch)
            # print('training loss', np.mean(loss_this_epoch))

            # this code from Rohan:
            # now, once an epoch is done:
            model.eval()
            y_pred = model(torch.Tensor(x_val))
            # print(y_pred.flatten().size(), torch.Tensor(y_valid).size())
            if loss_type == "no_var_loss":
                loss = lossFn(y_pred.flatten(), torch.Tensor(y_val)).item()
            if loss_type == "var_loss":
                loss = lossFn(
                    y_pred[:, 0].flatten(),
                    torch.Tensor(y_val),
                    y_pred[:, 1].flatten(),
                ).item()
            if loss_type == "bnll_loss":
                loss = lossFn(y_pred[:, 0].flatten(),
                              y_pred[:, 1].flatten(),
                              torch.Tensor(y_val),
                              beta=beta_epoch
                              ).item()

            loss_validation.append(loss)
            mse_loss = torch.nn.MSELoss(reduction='mean')
            mse = mse_loss(y_pred[:,0], torch.Tensor(y_val)).item()
        
            if loss < best_loss:
                best_loss = loss
                if verbose:
                    print("new best loss", loss, "in epoch", epoch)
                # best_weights = copy.deepcopy(model.state_dict())
            # print('validation loss', mse)
            if (plot or savefig) and (e % (EPOCHS-1) == 0) and (e != 0):
                ax1.plot(range(0, 1000),
                         range(0, 1000),
                         color='black',
                         ls='--')
                if loss_type == "no_var_loss":
                    ax1.scatter(
                        y_val,
                        y_pred.flatten().detach().numpy(),
                        color="#F45866",
                        edgecolor="black",
                        zorder=100,
                        label='validation dtata'
                    )
                else:
                    ax1.errorbar(
                        y_val,
                        y_pred[:, 0].flatten().detach().numpy(),
                        yerr=np.sqrt(y_pred[:, 1].
                                    flatten().detach().numpy()),
                        linestyle="None",
                        color="black",
                        capsize=2,
                        zorder=100,
                    )
                    ax1.scatter(
                        y_val,
                        y_pred[:, 0].flatten().detach().numpy(),
                        color="#9CD08F",
                        s=5,
                        zorder=101,
                        label='validation data'
                    )
                   
                # add residual plot
                residuals = y_pred[:, 0].flatten().detach().numpy() - y_val
                ax2.errorbar(y_val, residuals,
                             yerr=np.sqrt(y_pred[:, 1].
                                    flatten().detach().numpy()),
                            linestyle="None",
                            color="black",
                            capsize=2)
                ax2.scatter(y_val, residuals,
                            color='#9B287B',
                            s=5,
                            zorder=100)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2.set_ylabel("Residuals")
                ax2.set_xlabel("True Value")
                # add annotion for loss value
                if loss_type == "bnll_loss":
                    ax1.annotate(r'$\beta = $' +
                                    str(round(beta_epoch, 2)) + '\n' + 
                                    str(loss_type) + ' = ' + str(round(loss,2)) + '\n' +
                                    r'MSE = ' + str(round(mse,2)),
                             xy=(0.73, 0.1),
                             xycoords='axes fraction',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.5))

                else:
                    ax1.annotate(str(loss_type) + ' = ' + str(round(loss,2)) + '\n' + r'MSE = ' + str(round(mse,2)),
                             xy=(0.73, 0.1),
                             xycoords='axes fraction',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.5))  
                ax1.set_ylabel("Prediction")
                ax1.set_title("Epoch " + str(e))
                ax1.set_xlim([0, 1000])
                ax1.set_ylim([0, 1000])
                ax1.legend()
                if savefig:
                    ax1.errorbar(200, 600, yerr=5,
                                    color='red', capsize=2)
                    plt.savefig("../images/animations/" +
                                str(model_name) + "_nmodel_" +
                                str(m) + "_beta_" + str(beta_epoch) +
                                "_epoch_" + str(epoch) + ".png")
                if plot:
                    plt.show()
                plt.close()

            if save_all_checkpoints:
                if loss_type == "bnll_loss":
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "train_loss": np.mean(loss_this_epoch),
                            "valid_loss": loss,
                            "valid_mse": mse,
                            "valid_mean": y_pred[:, 0].flatten(),
                            "valid_sigma": y_pred[:, 1].flatten(),
                            "x_val": x_val,
                            "y_val": y_val,
                        },
                        path_to_model + "/" +
                        str(model_name) + "_beta_" + str(beta_epoch) +
                        "_nmodel_" + str(m) +
                        "_epoch_" + str(epoch) + ".pt",
                    )
                else:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "train_loss": np.mean(loss_this_epoch),
                            "valid_loss": loss,
                            "valid_mse": mse,
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
            if save_final_checkpoint and (e % (EPOCHS-1) == 0) and (e != 0):
                # option to just save final epoch
                if loss_type == "bnll_loss":
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "train_loss": np.mean(loss_this_epoch),
                            "valid_loss": loss,
                            "valid_mse": mse,
                            "valid_mean": y_pred[:, 0].flatten(),
                            "valid_sigma": y_pred[:, 1].flatten(),
                            "x_val": x_val,
                            "y_val": y_val,
                        },
                        path_to_model + "/" +
                        str(model_name) + "_beta_" + str(beta_epoch) +
                        "_nmodel_" + str(m) +
                        "_epoch_" + str(epoch) + ".pt",
                    )
                else:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "train_loss": np.mean(loss_this_epoch),
                            "valid_loss": loss,
                            "valid_mse": mse,
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
        final_mse.append(mse)

    endTime = time.time()
    if verbose:
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
