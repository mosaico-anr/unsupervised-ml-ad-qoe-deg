# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import logging, sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from src.utils.algorithm_utils import PyTorchUtils, AverageMeter

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



class DeepSVDDModel(nn.Module, PyTorchUtils):
    """
    Follows SKLearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.decision_function)
    """

    def __init__(self, D: int, seed: int, gpu: int):
        """Deep-SVDD model architecture.

        Args:
            D (int)                 : 
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.D = D
        self.net = self._build_network()
        self.rep_dim = D // 4

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.D, self.D // 2),
            nn.Tanh(),
            nn.Linear(self.D // 2, self.D // 4)
        )

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        return {'D': self.D, 'rep_dim': self.rep_dim}

def fit_with_early_stopping(train_loader, val_loader, model, patience, num_epochs, lr,
                            writer, center=None, R=0.0, objective='soft', verbose=True):
    """The fitting function of the Deep-SVDD.

    Args:
        train_loader (Dataloader)       : The train dataloader.
        val_loader (Dataloader)         : The val dataloader.
        model (nn.Module)               : The Pytorch model.
        patience (int)                  : The number of epochs to wait for early stopping.
        num_epochs (int)                : The max number of epochs.
        lr (float)                      : The learning rate.
        writer (SummaryWriter)          : The Tensorboard Summary Writer.
        center (torch.Tensor, optional) : Hypersphere center. Default to None.
        R (float, optional)             : Hypersphere radius. Defaults to 0.0.
        objective (str, optional)       : Objective function to use. Defaults to soft.
        verbose (bool, optional)        : Defaults to True.

    Returns:
                        [nn.Module ]: The fitted model.
    """
    model.to(model.device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-16)
    
    model.train()
    #train_loss_by_epoch = []
    #val_loss_by_epoch = []
    best_val_loss = np.inf
    epoch_wo_improv = 0
    best_params = model.state_dict()
    # assuming first batch is complete

    # Initialize hypersphere center c with train loader
    if center is None:
        print("Initializing center c...")
        center = init_center_c(model, train_loader)
        print(center)
        print("Center c initialized.")
        
    # Initialize the radius
    R = torch.tensor(R, device=model.device)
    update_each = 5
    nu = 0.1
    # objective = 'soft'

    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            #if verbose:
                #GPUtil.showUtilization()
            # Train the model
            #logger.debug("Begin training...")
            train_loss, R = train(train_loader, model, optimizer, epoch, center, R,
                                  objective, update_each, nu)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader, model, optimizer, epoch, center, R, objective, nu)
            
            if verbose:
                logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.2f} - Val loss: {val_loss:.2f}")
            
            # Write in TensorBoard
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            
            # Check if the validation loss improve or not
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                epoch_wo_improv = 0
                best_params = model.state_dict()
            elif val_loss >= best_val_loss:
                epoch_wo_improv += 1
            
        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break

    return model, center, R

def train(train_loader, model, optimizer, epoch, center, R, objective='soft', update_each=5, nu=0.1):
    """The training step.


    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        center (torch.Tensor)           : Hypersphere center.
        R (float)                       : Hypersphere radius.
        objective (str, optional)       : Objective function to use. Defaults to 'soft'.
        update_each (int, optional)     : Frequency to update radius. Defaults to 5.
        nu (float, optional)            : Deep-SVDD hyperparameters on outliers. Defaults to 0.1.

    Returns:
        torch.Tensor                    : Training loss.
    """
    # Compute statistics
    loss_meter = AverageMeter()
    
    #
    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)
        output = model(ts_batch)

        dist = torch.sum((output - center)**2, dim=1)
        if objective=='soft':
            scores = dist - R **2
            loss = R**2 + (1/nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
            
        model.zero_grad()
        loss.backward()
        optimizer.step()
            
        # Update radius
        if (objective=='soft') and (epoch >=update_each):
            R.data = torch.tensor(get_radius(dist, nu), device=model.device)

        
        # multiplying by length of batch to correct accounting for incomplete batches
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg, R
    
def validation(val_loader, model, optimizer, epoch, center, R, objective='soft', nu=0.1):
    """The validation step.


    Args:
        val_loader (Dataloader)         : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        center (torch.Tensor)           : Hypersphere center.
        R (float)                       : Hypersphere radius.
        objective (str, optional)       : Objective function to use. Defaults to 'soft'.
        nu (float, optional)            : Deep-SVDD hyperparameters on outliers. Defaults to 0.1.

    Returns:
        torch.Tensor                    : Validation loss.
    """

    # Compute statistics
    loss_meter = AverageMeter()
    
    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)
            output = model(ts_batch)

            dist = torch.sum((output - center)**2, dim=1)
            if objective=='soft':
                scores = dist - R **2
                loss = R**2 + (1/nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss_meter.update(loss.item())
        return loss_meter.avg

@torch.no_grad()
def predict_test_scores(model, test_loader, center, R, objective):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        .
    Returns:
                The reconstruction score 

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader
        center (torch.Tensor)           : Hypersphere center.
        R (float)                       : Hypersphere radius.
        objective (str)                 : Objective function to use.

    Returns:
        _type_: The reconstruction score
    """
    model.eval()
    scores = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(model.device)
        output = model(ts_batch)
        dist = torch.sum((output - center)**2, dim=1)
        if objective=='soft':
            score = dist - R **2
        else:
            score = dist
        scores.extend(score.cpu().tolist())

    return np.array(scores)

def init_center_c(model, train_loader: DataLoader, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data.
        Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
    n_samples = 0
    c = torch.zeros(model.rep_dim, device=model.device)

    model.eval()
    with torch.no_grad():
        for sample in train_loader:
            # get the inputs of the batch
            X = sample.float().to(model.device)
            outputs = model(X)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1-nu)

