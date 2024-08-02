import math
import torch
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from models import models


def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)


def train_DER(
    trainDataLoader,
    x_val,
    y_val,
    INIT_LR,
    DEVICE,
    COEFF,
    loss_type,
    norm_params: dict,
    model_name="DER",
    EPOCHS=100,
    path_to_model="models/",
    data_prescription="linear_homoskedastic",
    inject_type="predictive",
    data_dim="0D",
    noise_level="low",
    save_all_checkpoints=False,
    save_final_checkpoint=False,
    overwrite_final_checkpoint=False,
    plot=True,
    savefig=True,
    set_and_save_rs=False,
    rs=42,
    save_n_hidden=False,
    n_hidden=64,
    save_size_df: bool = False,
    size_df: float = 10000,
    verbose=True,
):
    # first determine if you even need to run anything
    if not save_all_checkpoints and save_final_checkpoint:
        # option to skip running the model if you don't care about
        # saving all checkpoints and only want to save the final
        final_chk = (
            str(path_to_model)
            + "checkpoints/"
            + str(model_name)
            + "_"
            + str(data_prescription)
            + "_"
            + str(inject_type)
            + "_noise_"
            + str(noise_level)
            + "_loss_"
            + str(loss_type)
            + "_epoch_"
            + str(EPOCHS - 1)
            + ".pt"
        )
        if verbose:
            print("final chk", final_chk)
            # check if the final epoch checkpoint already exists
            print(glob.glob(final_chk))
        if glob.glob(final_chk):
            print("final model already exists")
            if overwrite_final_checkpoint:
                print("going to overwrite final checkpoint")
            else:
                print("not overwriting, exiting")
                return
        else:
            print("model does not exist yet, going to save")
    # measure how long training is going to take
    if verbose:
        print("[INFO] training the network...")
        print("saving all checkpoints?")
        print(save_all_checkpoints)
        print("saving final checkpoint?")
        print(save_final_checkpoint)
        print("overwriting final checkpoint if its already there?")
        print(overwrite_final_checkpoint)
        print(f"saving here: {path_to_model}")
        print(f"model name: {model_name}")

    startTime = time.time()
    start_epoch = 0

    if set_and_save_rs:
        print("setting and saving the rs")
        # Set the random seed
        set_random_seeds(seed_value=rs)

    best_loss = np.inf  # init to infinity
    model, lossFn = models.model_setup_DER(
        loss_type, DEVICE, n_hidden=n_hidden, data_type=data_dim
    )
    if verbose:
        print("model is", model, "lossfn", lossFn)
    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    mse_loss = torch.nn.MSELoss(reduction="mean")

    # loop over our epochs
    for e in range(0, EPOCHS):
        if plot or savefig:
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
            )

        epoch = int(start_epoch + e)

        # set the model in training mode
        model.train()

        # loop over the training set
        if verbose:
            print("epoch", epoch, round(e / EPOCHS, 2))
        loss_this_epoch = []
        mse_this_epoch = []
        # randomly shuffles the training data (if shuffle = True)
        # and draws batches up to the total training size
        # (should be about 8 batches)
        for i, (x, y) in enumerate(trainDataLoader):
            # print('i', i, len(y))
            # send the input to the device
            # (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            print('pred', pred[:, 0].flatten().detach().numpy())

            plt.clf()
            plt.scatter(
                x[:,0].detach().numpy(),
                pred[:, 0].flatten().detach().numpy(),
                label='pred')
            plt.scatter(
                x[:,0].detach().numpy(), y.detach().numpy(),
                label='true')
            plt.annotate('batch '+str(i),
                         xy=(0.02, 0.9),
                         xycoords='axes fraction')
            plt.legend()
            plt.show()
            
            
            loss = lossFn(pred, y, COEFF)
            if plot or savefig:
                if (e % (EPOCHS - 1) == 0) and (e != 0):
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
                """
                else:
                    ax1.scatter(y,
                                pred[:, 0].flatten().detach().numpy(),
                                color="grey")
                """
            loss_this_epoch.append(loss[0].item())
            mse_this_epoch.append(mse_loss(pred[:, 0], y).item())
            # zero out the gradients
            opt.zero_grad()
            # perform the backpropagation step
            # computes the derivative of loss with respect to the parameters
            loss[0].backward()
            # update the weights
            # optimizer takes a step based on the gradients of the parameters
            # here, its taking a step for every batch
            opt.step()
        model.eval()
        y_pred = model(torch.Tensor(x_val))
        loss = lossFn(y_pred, torch.Tensor(y_val), COEFF)
        NIGloss_val = loss[0].item()
        assert not math.isnan(NIGloss_val), \
            f"loss is: {loss}, terminating training"
        mean_u_al_val = np.mean(loss[1])
        mean_u_ep_val = np.mean(loss[2])
        std_u_al_val = np.std(loss[1])
        std_u_ep_val = np.std(loss[2])

        # lets also grab mse loss
        mse = mse_loss(y_pred[:, 0], torch.Tensor(y_val)).item()
        if NIGloss_val < best_loss:
            best_loss = NIGloss_val
            if verbose:
                print("new best loss", NIGloss_val, "in epoch", epoch)
                print("meanwhile mse is", mse)
            # best_weights = copy.deepcopy(model.state_dict())
        if (plot or savefig) and (e != 0) and (e % (EPOCHS - 1) == 0):
            # ax1.plot(range(0, 1000), range(0, 1000), color="black", ls="--")
            if loss_type == "no_var_loss":
                ax1.scatter(
                    y_val,
                    y_pred.flatten().detach().numpy(),
                    color="#F45866",
                    edgecolor="black",
                    zorder=100,
                    label="validation dtata",
                )
            else:
                ax1.errorbar(
                    y_val,
                    y_pred[:, 0].flatten().detach().numpy(),
                    yerr=np.sqrt(y_pred[:, 1].flatten().detach().numpy()),
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
                    label="validation data",
                )

            # add residual plot
            residuals = y_pred[:, 0].flatten().detach().numpy() - y_val
            ax2.errorbar(
                y_val,
                residuals,
                yerr=np.sqrt(y_pred[:, 1].flatten().detach().numpy()),
                linestyle="None",
                color="black",
                capsize=2,
            )
            ax2.scatter(y_val, residuals, color="#9B287B", s=5, zorder=100)
            ax2.axhline(0, color="black", linestyle="--", linewidth=1)
            ax2.set_ylabel("Residuals")
            ax2.set_xlabel("True Value")
            # add annotion for loss value
            ax1.annotate(
                str(loss_type)
                + " = "
                + str(round(NIGloss_val, 2))
                + "\n"
                + r"MSE = "
                + str(round(mse, 2)),
                xy=(0.73, 0.1),
                xycoords="axes fraction",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="lightgrey",
                    alpha=0.5),
            )
            ax1.set_ylabel("Prediction")
            ax1.set_title("Epoch " + str(e))
            # ax1.set_xlim([0, 1000])
            # ax1.set_ylim([0, 1000])
            ax1.legend()
            if savefig:
                # ax1.errorbar(200, 600, yerr=5,
                #                color='red', capsize=2)
                plt.savefig(
                    str(path_to_model)
                    + "images/animations/"
                    + str(model_name)
                    + "_"
                    + str(data_prescription)
                    + "_"
                    + str(inject_type)
                    + "_"
                    + str(data_dim)
                    + "_loss_"
                    + str(loss_type)
                    + "_COEFF_"
                    + str(COEFF)
                    + "_epoch_"
                    + str(epoch)
                    + ".png"
                )
            if plot:
                plt.show()
            plt.close()
        if save_all_checkpoints:

            filename = (
                str(path_to_model)
                + "checkpoints/"
                + str(model_name)
                + "_"
                + str(data_prescription)
                + "_"
                + str(inject_type)
                + "_"
                + str(data_dim)
                + "_noise_"
                + str(noise_level)
                + "_loss_"
                + str(loss_type)
                + "_COEFF_"
                + str(COEFF)
                + "_epoch_"
                + str(epoch)
            )
            if set_and_save_rs:
                filename += "_rs_" + str(rs)
            if save_n_hidden:
                filename += "_n_hidden_" + str(n_hidden)
            if save_size_df:
                filename += "_sizedf_" + str(size_df)
            filename += ".pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": np.mean(loss_this_epoch),
                    "valid_loss": NIGloss_val,
                    "train_mse": np.mean(mse_this_epoch),
                    "valid_mse": mse,
                    "mean_u_al_validation": mean_u_al_val,
                    "mean_u_ep_validation": mean_u_ep_val,
                    "std_u_al_validation": std_u_al_val,
                    "std_u_ep_validation": std_u_ep_val,
                    "norm_params": norm_params,
                },
                filename,
            )
            if epoch == 99:
                print("checkpoint saved here", filename)

        if save_final_checkpoint and (e % (EPOCHS - 1) == 0) and (e != 0):
            filename = (
                str(path_to_model)
                + "checkpoints/"
                + str(model_name)
                + "_"
                + str(data_prescription)
                + "_"
                + str(inject_type)
                + "_"
                + str(data_dim)
                + "_noise_"
                + str(noise_level)
                + "_loss_"
                + str(loss_type)
                + "_COEFF_"
                + str(COEFF)
                + "_epoch_"
                + str(epoch)
            )
            if set_and_save_rs:
                filename += "_rs_" + str(rs)
            if save_n_hidden:
                filename += "_n_hidden_" + str(n_hidden)
            if save_size_df:
                filename += "_sizedf_" + str(size_df)
            filename += ".pt"
            # option to just save final epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": np.mean(loss_this_epoch),
                    "valid_loss": NIGloss_val,
                    "train_mse": np.mean(mse_this_epoch),
                    "valid_mse": mse,
                    "mean_u_al_validation": mean_u_al_val,
                    "mean_u_ep_validation": mean_u_ep_val,
                    "std_u_al_validation": std_u_al_val,
                    "std_u_ep_validation": std_u_ep_val,
                    "norm_params": norm_params,
                },
                filename,
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
    INIT_LR: float,
    DEVICE,
    loss_type: str,
    n_models: float,
    norm_params: dict,
    model_name: str = "DE",
    BETA: float = 0.5,
    EPOCHS: float = 100,
    path_to_model: str = "models/",
    data_prescription: str = "linear_homoskedastic",
    inject_type: str = "predictive",
    data_dim: str = "0D",
    noise_level: str = "low",
    save_all_checkpoints: bool = False,
    save_final_checkpoint: bool = False,
    overwrite_final_checkpoint: bool = False,
    plot: bool = True,
    savefig: bool = True,
    set_and_save_rs: bool = False,
    rs_list: list[int] = [42, 42],
    save_n_hidden: bool = False,
    n_hidden: float = 64,
    save_size_df: bool = False,
    size_df: float = 10000,
    verbose: bool = True,
):

    startTime = time.time()
    start_epoch = 0
    if verbose:
        print("starting here", start_epoch)

    loss_all_epochs = []  # this is from the training set
    mse_all_epochs = []
    loss_validation = []
    final_mse = []

    best_loss = np.inf  # init to infinity

    model_ensemble = []

    for m in range(n_models):
        print("model", m)
        if not save_all_checkpoints and save_final_checkpoint:
            # option to skip running this model if you don't care about
            # saving all checkpoints and only want to save the final
            if loss_type == "bnll_loss":
                final_chk = (
                    str(path_to_model)
                    + "checkpoints/"
                    + str(model_name)
                    + "_"
                    + str(data_prescription)
                    + "_"
                    + str(inject_type)
                    + "_"
                    + str(data_dim)
                    + "_noise_"
                    + str(noise_level)
                    + "_beta_"
                    + str(BETA)
                    + "_nmodel_"
                    + str(m)
                    + "_epoch_"
                    + str(EPOCHS - 1)
                    + ".pt"
                )
            else:
                final_chk = (
                    str(path_to_model)
                    + "checkpoints/"
                    + str(model_name)
                    + "_"
                    + str(data_prescription)
                    + "_"
                    + str(inject_type)
                    + "_"
                    + str(data_dim)
                    + "_noise_"
                    + str(noise_level)
                    + "_nmodel_"
                    + str(m)
                    + "_epoch_"
                    + str(EPOCHS - 1)
                    + ".pt"
                )
            if verbose:
                print("final chk", final_chk)
                # check if the final epoch checkpoint already exists
                print(glob.glob(final_chk))
            if glob.glob(final_chk):
                print("final model already exists")
                if overwrite_final_checkpoint:
                    print("going to overwrite final checkpoint")
                else:
                    print("not overwriting, skipping to next model in loop")
                    continue
            else:
                print("model does not exist yet, going to save")
        if set_and_save_rs:
            assert (
                len(rs_list) == n_models
            ), "you are attempting to use the random seed list \
                  but the lengths don't match"
            rs = rs_list[m]
            print("setting and saving the rs")
            # Set the random seed
            set_random_seeds(seed_value=rs)
        # initialize the model again each time from scratch
        model, lossFn = models.model_setup_DE(
            loss_type, DEVICE, n_hidden=n_hidden, data_type=data_dim
        )
        if verbose:
            print("model is", model, "lossfn", lossFn)
        opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
        mse_loss = torch.nn.MSELoss(reduction="mean")

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
            mse_this_epoch = []
            if plot or savefig:
                plt.clf()
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
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
                                  y, pred[:, 1].flatten())
                if loss_type == "bnll_loss":
                    """
                    if e/EPOCHS < 0.2:
                        # use beta = 1
                        beta_epoch = 1
                    if (e/EPOCHS > 0.2) & (e/EPOCHS < 0.5):
                        beta_epoch = 0.75
                    if e/EPOCHS > 0.5:
                        beta_epoch = 0.5
                    # 1 - e / EPOCHS # this one doesn't work great
                    """
                    if BETA == "linear_decrease":
                        beta_epoch = 1 - e / EPOCHS
                    if BETA == "step_decrease_to_0.5":
                        if e / EPOCHS < 0.5:
                            beta_epoch = 1
                        else:
                            beta_epoch = 0.5
                    if BETA == "step_decrease_to_0.0":
                        if e / EPOCHS < 0.5:
                            beta_epoch = 1
                        else:
                            beta_epoch = 0.0

                    # Try to convert the BETA input to a constant float value
                    try:
                        beta_epoch = float(BETA)
                    except ValueError:
                        pass
                    loss = lossFn(
                        pred[:, 0].flatten(), pred[:, 1].flatten(),
                        y, beta=beta_epoch
                    )
                    mse = mse_loss(pred[:, 0], y)
                if plot or savefig:
                    if (e % (EPOCHS - 1) == 0) and (e != 0):
                        if loss_type == "no_var_loss":
                            ax1.scatter(
                                y,
                                pred.flatten().detach().numpy(),
                                color="grey",
                                alpha=0.5,
                                label="training data",
                            )
                        else:
                            if i == 0:
                                ax1.scatter(
                                    y,
                                    pred[:, 0].flatten().detach().numpy(),
                                    color="grey",
                                    alpha=0.5,
                                    label="training data",
                                )
                            else:
                                ax1.scatter(
                                    y,
                                    pred[:, 0].flatten().detach().numpy(),
                                    color="grey",
                                    alpha=0.5,
                                )
                loss_this_epoch.append(loss.item())
                mse_this_epoch.append(mse.item())

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
            mse_all_epochs.append(mse_this_epoch)
            # print('training loss', np.mean(loss_this_epoch))

            # this code from Rohan:
            # now, once an epoch is done:
            model.eval()
            y_pred_val = model(torch.Tensor(x_val))
            # print(y_pred.flatten().size(), torch.Tensor(y_valid).size())
            if loss_type == "no_var_loss":
                loss_val = lossFn(y_pred_val.flatten(),
                                  torch.Tensor(y_val)).item()
            if loss_type == "var_loss":
                loss_val = lossFn(
                    y_pred_val[:, 0].flatten(),
                    torch.Tensor(y_val),
                    y_pred_val[:, 1].flatten(),
                ).item()
            if loss_type == "bnll_loss":
                loss_val = lossFn(
                    y_pred_val[:, 0].flatten(),
                    y_pred_val[:, 1].flatten(),
                    torch.Tensor(y_val),
                    beta=beta_epoch,
                ).item()
            assert not math.isnan(loss_val), \
                f"loss is: {loss_val}, terminating training"
            loss_validation.append(loss_val)
            mse = mse_loss(y_pred_val[:, 0], torch.Tensor(y_val)).item()
            if loss_val < best_loss:
                best_loss = loss_val
                if verbose:
                    print("new best loss", loss_val, "in epoch", epoch)
                # best_weights = copy.deepcopy(model.state_dict())
            if (plot or savefig) and (e % (EPOCHS - 1) == 0) and (e != 0):
                ax1.plot(range(0, 1000), range(0, 1000),
                         color="black", ls="--")
                if loss_type == "no_var_loss":
                    ax1.scatter(
                        y_val,
                        y_pred_val.flatten().detach().numpy(),
                        color="#F45866",
                        edgecolor="black",
                        zorder=100,
                        label="validation dtata",
                    )
                else:
                    ax1.errorbar(
                        y_val,
                        y_pred_val[:, 0].flatten().detach().numpy(),
                        yerr=np.sqrt(
                            y_pred_val[:, 1].flatten().detach().numpy()),
                        linestyle="None",
                        color="black",
                        capsize=2,
                        zorder=100,
                    )
                    ax1.scatter(
                        y_val,
                        y_pred_val[:, 0].flatten().detach().numpy(),
                        color="#9CD08F",
                        s=5,
                        zorder=101,
                        label="validation data",
                    )
                    ax1.scatter(
                        y,
                        pred[:, 0].flatten().detach().numpy(),
                        color="red",
                        s=5,
                        zorder=101,
                        label="training data",
                    )

                # add residual plot
                residuals = y_pred_val[:, 0].flatten().detach().numpy() - y_val
                ax2.errorbar(
                    y_val,
                    residuals,
                    yerr=np.sqrt(y_pred_val[:, 1].flatten().detach().numpy()),
                    linestyle="None",
                    color="black",
                    capsize=2,
                )
                ax2.scatter(y_val, residuals, color="#9B287B", s=5, zorder=100)
                ax2.axhline(0, color="black", linestyle="--", linewidth=1)
                ax2.set_ylabel("Residuals")
                ax2.set_xlabel("True Value")
                # add annotion for loss value
                if loss_type == "bnll_loss":
                    ax1.annotate(
                        r"$\beta = $"
                        + str(round(beta_epoch, 2))
                        + "\n"
                        + str(loss_type)
                        + " = "
                        + str(round(loss_val, 2))
                        + "\n"
                        + r"MSE = "
                        + str(round(mse, 2)),
                        xy=(0.73, 0.1),
                        xycoords="axes fraction",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="lightgrey",
                            alpha=0.5
                        ),
                    )

                else:
                    ax1.annotate(
                        str(loss_type)
                        + " = "
                        + str(round(loss, 2))
                        + "\n"
                        + r"MSE = "
                        + str(round(mse, 2)),
                        xy=(0.73, 0.1),
                        xycoords="axes fraction",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="lightgrey",
                            alpha=0.5
                        ),
                    )
                ax1.set_ylabel("Prediction")
                ax1.set_title("Epoch " + str(e))
                ax1.set_xlim([0, 1000])
                ax1.set_ylim([0, 1000])
                ax1.legend()
                if savefig:
                    # ax1.errorbar(200, 600, yerr=5,
                    #                color='red', capsize=2)
                    plt.savefig(
                        str(path_to_model)
                        + "images/animations/"
                        + str(model_name)
                        + "_"
                        + str(data_prescription)
                        + "_"
                        + str(inject_type)
                        + "_"
                        + str(data_dim)
                        + "_noise_"
                        + str(noise_level)
                        + "_nmodel_"
                        + str(m)
                        + "_beta_"
                        + str(BETA)
                        + "_epoch_"
                        + str(epoch)
                        + ".png"
                    )
                if plot:
                    plt.show()
                plt.close()

            if save_all_checkpoints:
                filename = (
                    str(path_to_model)
                    + "checkpoints/"
                    + str(model_name)
                    + "_"
                    + str(data_prescription)
                    + "_"
                    + str(inject_type)
                    + "_"
                    + str(data_dim)
                    + "_noise_"
                    + str(noise_level)
                )
                if loss_type == "bnll_loss":
                    filename += "_beta_" + str(BETA)
                filename += "_nmodel_" + str(m) + "_epoch_" + str(epoch)
                if set_and_save_rs:
                    filename += "_rs_" + str(rs)
                if save_n_hidden:
                    filename += "_n_hidden_" + str(n_hidden)
                if save_size_df:
                    filename += "_sizedf_" + str(size_df)
                filename += ".pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": np.mean(loss_this_epoch),
                        "valid_loss": loss_val,
                        "train_mse": np.mean(mse_this_epoch),
                        "valid_mse": mse,
                        "valid_mean": y_pred_val[:, 0].flatten(),
                        "valid_var": y_pred_val[:, 1].flatten(),
                        "x_val": x_val,
                        "y_val": y_val,
                        "norm_params": norm_params,
                    },
                    filename,
                )
            if save_final_checkpoint and (e % (EPOCHS - 1) == 0) and (e != 0):
                # option to just save final epoch
                filename = (
                    str(path_to_model)
                    + "checkpoints/"
                    + str(model_name)
                    + "_"
                    + str(data_prescription)
                    + "_"
                    + str(inject_type)
                    + "_"
                    + str(data_dim)
                    + "_noise_"
                    + str(noise_level)
                )
                if loss_type == "bnll_loss":
                    filename += "_beta_" + str(BETA)
                filename += "_nmodel_" + str(m) + "_epoch_" + str(epoch)
                if set_and_save_rs:
                    filename += "_rs_" + str(rs)
                if save_n_hidden:
                    filename += "_n_hidden_" + str(n_hidden)
                if save_size_df:
                    filename += "_sizedf_" + str(size_df)
                filename += ".pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": np.mean(loss_this_epoch),
                        "valid_loss": loss_val,
                        "train_mse": np.mean(mse_this_epoch),
                        "valid_mse": mse,
                        "valid_mean": y_pred_val[:, 0].flatten(),
                        "valid_var": y_pred_val[:, 1].flatten(),
                        "x_val": x_val,
                        "y_val": y_val,
                        "norm_params": norm_params,
                    },
                    filename,
                )
                print('saved final checkpoint', filename)
        model_ensemble.append(model)
        final_mse.append(mse)

    endTime = time.time()
    if verbose:
        print("start at", startTime, "end at", endTime)
        print(endTime - startTime)

    return model_ensemble
