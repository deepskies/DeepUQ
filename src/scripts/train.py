"""
Simple stubs to use for re-train of the final model
Can leave a default data source, or specify that 'load data' loads the dataset used in the final version
"""
import argparse
import torch
import sbi
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.scripts import models
import functools


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

def train_DER(trainDataLoader,
              x_val,
              y_val,
              INIT_LR,
              DEVICE,
              COEFF,
              DER_type,
              model_name='DER',
              EPOCHS=40,
              save_checkpoints=False,
              plot=False):
    # measure how long training is going to take
    print("[INFO] training the network...")

    print("saving checkpoints?")
    print(save_checkpoints)

    startTime = time.time()

    # Find last epoch saved
    if save_checkpoints:
        
        print(glob.glob('models/*'+model_name+'*'))
        list_models_run = []
        for file in glob.glob('models/*'+model_name+'*'):
            list_models_run.append(float(str.split(str(str.split(file, model_name+'_')[1]),'.')[0]))
        if list_models_run:
            start_epoch = max(list_models_run) + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    print('starting here', start_epoch)


    loss_all_epochs = [] # this is from the training set
    loss_validation = []

    best_loss = np.inf   # init to infinity


    # initialize the model from scratch
    if DER_type == 'SDER':
        #model = models.de_no_var().to(device)
        DERLayer = models.SDERLayer
    
        # initialize our loss function
        lossFn = models.loss_sder
    else:
        #model = models.de_var().to(device)
        DERLayer = models.DERLayer
        # initialize our loss function
        lossFn = models.loss_der
    
    # from https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/x3_indepth.ipynb 
    model = torch.nn.Sequential(models.Model(4), DERLayer())
    model = model.to(DEVICE)
    loss_fct = functools.partial(lossFn, coeff=COEFF)
    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # loop over our epochs
    for e in range(0, EPOCHS):
        epoch = int(start_epoch + e)

        # set the model in training mode
        model.train()

        # loop over the training set
        print('epoch', epoch, round(e/EPOCHS,2))

        loss_this_epoch = []

        plt.clf()
        # randomly shuffles the training data (if shuffle = True)
        # and draws batches up to the total training size
        # (should be about 8 batches)
        for i, (x, y) in enumerate(trainDataLoader):
            #print('i', i, len(y))
            # send the input to the device
            #(x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss

            pred = model(x)
            print('shapes train', np.shape(pred), np.shape(y))
        
            loss = lossFn(pred, y, COEFF)
            if plot == True:
                if e % 5 == 0:
                    if i == 0:
                        #if loss_type == 'no_var_loss':
                        plt.scatter(y, pred[:, 0].flatten().detach().numpy(),
                                    color='#F45866',
                                    edgecolor='black',
                                    zorder=100)
                        '''
                        else:
                            plt.errorbar(y,
                                            pred[:, 0].flatten().detach().numpy(),
                                            yerr=abs(pred[:, 1].flatten().detach().numpy()),
                                            linestyle='None',
                                            color='#F45866',
                                            zorder=100)
                            plt.scatter(y,
                                        pred[:, 0].flatten().detach().numpy(),
                                        color='#F45866',
                                        edgecolor='black',
                                        zorder=100)
                        '''
                    else:
                        plt.scatter(y, pred[:, 0].flatten().detach().numpy())
                        
            loss_this_epoch.append(loss.item())

            # zero out the gradients        
            opt.zero_grad()
            # perform the backpropagation step
            # computes the derivative of loss with respect to the parameters
            loss.backward()
            # update the weights
            # optimizer takes a step based on the gradients of the parameters
            # here, its taking a step for every batch
            opt.step()
        if plot == True:
            if e % 5 == 0:    
                plt.ylabel('prediction')
                plt.xlabel('true value')
                plt.title('Epoch '+str(e))
                plt.show()   
        loss_all_epochs.append(loss_this_epoch)
        #print('training loss', np.mean(loss_this_epoch))

        # this code from Rohan:
        #now, once an epoch is done:
        model.eval()
        y_pred = model(torch.Tensor(x_val))
        print('shapes val', np.shape(y_pred), np.shape(y))
        NIGloss_val = lossFn(y_pred, y, COEFF).item()

        loss_validation.append(NIGloss_val)
        if NIGloss_val < best_loss:
            best_loss = NIGloss_val
            print('new best loss', NIGloss_val, 'in epoch', epoch)
            #best_weights = copy.deepcopy(model.state_dict())
        #print('validation loss', mse)

        if save_checkpoints:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': np.mean(loss_this_epoch),
                'valid_loss': mse
                }, "/home/rnevin/deepskieslab/rnevin/TinyCNN/models/TinyCNN_MSE_"+str(epoch)+".pt")
    endTime = time.time()
    print('start at', startTime, 'end at', endTime)
    print(endTime - startTime)

    return model

def train_DE(trainDataLoader,
             x_val,
             y_val,
             INIT_LR,
             device,
             loss_type,
             n_models,
             model_name='DE',
             EPOCHS=40,
             save_checkpoints=False,
             plot=False):
    # measure how long training is going to take
    print("[INFO] training the network...")

    print("saving checkpoints?")
    print(save_checkpoints)

    startTime = time.time()

    # Find last epoch saved
    if save_checkpoints:
        
        print(glob.glob('models/*'+model_name+'*'))
        list_models_run = []
        for file in glob.glob('models/*'+model_name+'*'):
            list_models_run.append(float(str.split(str(str.split(file, model_name+'_')[1]),'.')[0]))
        if list_models_run:
            start_epoch = max(list_models_run) + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    print('starting here', start_epoch)


    loss_all_epochs = [] # this is from the training set
    loss_validation = []

    best_mse = np.inf   # init to infinity


    model_ensemble = []

    for m in range(n_models):
        # initialize the model again each time from scratch
        if loss_type == 'no_var_loss':
            model = models.de_no_var().to(device)
            # initialize our optimizer and loss function
            opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
            lossFn = torch.nn.MSELoss(reduction="mean")
        else:
            model = models.de_var().to(device)
            # initialize our optimizer and loss function
            opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
            lossFn = torch.nn.GaussianNLLLoss(full=False, eps=1e-06, reduction="mean")

        

        # loop over our epochs
        for e in range(0, EPOCHS):
            epoch = int(start_epoch + e)

            # set the model in training mode
            model.train()

            # loop over the training set
            print('epoch', epoch, round(e/EPOCHS,2))

            loss_this_epoch = []

            plt.clf()
            # randomly shuffles the training data (if shuffle = True)
            # and draws batches up to the total training size
            # (should be about 8 batches)
            for i, (x, y) in enumerate(trainDataLoader):
                #print('i', i, len(y))
                # send the input to the device
                #(x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss

                pred = model(x)
                if loss_type == 'no_var_loss':
                    loss = lossFn(pred.flatten(), y)
                else:
                    loss = lossFn(pred[:,0].flatten(), y, pred[:,1].flatten()**2)
                if plot == True:
                    if e % 5 == 0:
                        if i == 0:
                            if loss_type == 'no_var_loss':
                                plt.scatter(y, pred.flatten().detach().numpy(),
                                            color='#F45866',
                                            edgecolor='black',
                                            zorder=100)
                            else:
                                plt.errorbar(y,
                                             pred[:, 0].flatten().detach().numpy(),
                                             yerr=abs(pred[:, 1].flatten().detach().numpy()),
                                             linestyle='None',
                                             color='#F45866',
                                             zorder=100)
                                plt.scatter(y,
                                            pred[:, 0].flatten().detach().numpy(),
                                            color='#F45866',
                                            edgecolor='black',
                                            zorder=100)
                        else:
                            if loss_type == 'no_var_loss':
                                plt.scatter(y, pred.flatten().detach().numpy())
                            else:
                                plt.scatter(y,
                                            pred[:, 0].flatten().detach().numpy())
                            
                loss_this_epoch.append(loss.item())

                # zero out the gradients        
                opt.zero_grad()
                # perform the backpropagation step
                # computes the derivative of loss with respect to the parameters
                loss.backward()
                # update the weights
                # optimizer takes a step based on the gradients of the parameters
                # here, its taking a step for every batch
                opt.step()
            if plot == True:
                if e % 5 == 0:    
                    plt.ylabel('prediction')
                    plt.xlabel('true value')
                    plt.title('Epoch '+str(e))
                    plt.show()   
            loss_all_epochs.append(loss_this_epoch)
            #print('training loss', np.mean(loss_this_epoch))

            # this code from Rohan:
            #now, once an epoch is done:
            model.eval()
            y_pred = model(torch.Tensor(x_val))
            
            '''
            print(np.shape(x_val))
            if plot == True:
                if e % 5 == 0:
                    plt.clf()
                    fig = plt.figure(figsize = (10,4))
                    ax = fig.add_subplot(131)
                    
                    ax.scatter(x_val[:,0], y_val, label = 'actual', color = '#4E937A', s = 10, zorder = 100)
                    
                    if loss_type == 'no_var_loss':
                        ax.scatter(x_val[:,0], y_pred.detach().numpy(),
                                        label = 'predicted', color = '#C7F2A7', s = 10)
                    else:
                        ax.errorbar(x_val[:,0], y_pred[:,0].detach().numpy(),
                                    yerr = abs(y_pred[:,1].detach().numpy()),
                                    label = 'predicted', color = '#C7F2A7', ls = 'None')
                        

                    plt.legend()
                    ax.set_xlabel('input x dimension 0')
                    ax.set_ylabel('prediction')
                    
                    ax = fig.add_subplot(132)
                    ax.scatter(x_val[:,1], y_val, label = 'actual', color = '#4E937A', s = 10, zorder = 100)

                    if loss_type == 'no_var_loss':
                        ax.scatter(x_val[:,1], y_pred.detach().numpy(),
                                        label = 'predicted', color = '#C7F2A7', s = 10)
                    else:
                        ax.errorbar(x_val[:,1], y_pred[:,0].detach().numpy(),
                                    yerr = abs(y_pred[:,1].detach().numpy()),
                                    label = 'predicted', color = '#C7F2A7', linestyle = 'None')

                    plt.legend()
                    ax.set_xlabel('input x dimension 1')
                    ax.set_ylabel('prediction')
                    
                    ax = fig.add_subplot(133)
                    ax.scatter(x_val[:,2], y_val, label = 'actual', color = '#4E937A', s = 10, zorder = 100)

                    if loss_type == 'no_var_loss':
                        ax.scatter(x_val[:,2], y_pred.detach().numpy(),
                                        label = 'predicted', color = '#C7F2A7', s = 10)
                    else:
                        ax.errorbar(x_val[:,2], y_pred[:,0].detach().numpy(),
                                    yerr = abs(y_pred[:,1].detach().numpy()),
                                    label = 'predicted', color = '#C7F2A7', ls = 'None')
                    plt.legend()
                    ax.set_xlabel('input x dimension 2')
                    ax.set_ylabel('precition')
                    plt.show()
            '''
            #print(y_pred.flatten().size(), torch.Tensor(y_valid).size())
            if loss_type == 'no_var_loss':
                mse = lossFn(y_pred.flatten(), torch.Tensor(y_val)).item()
            else:
                mse = lossFn(y_pred[:,0].flatten(), torch.Tensor(y_val), y_pred[:,1].flatten()**2).item()

            loss_validation.append(mse)
            if mse < best_mse:
                best_mse = mse
                print('new best mse', mse, 'in epoch', epoch)
                #best_weights = copy.deepcopy(model.state_dict())
            #print('validation loss', mse)

            if save_checkpoints:

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': np.mean(loss_this_epoch),
                    'valid_loss': mse
                    }, "/home/rnevin/deepskieslab/rnevin/TinyCNN/models/TinyCNN_MSE_"+str(epoch)+".pt")

        model_ensemble.append(model)

        
        

    endTime = time.time()
    print('start at', startTime, 'end at', endTime)
    print(endTime - startTime)

    return model_ensemble



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
