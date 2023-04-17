"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import logging
import sys

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
    Deep-SVDD Model.
    Follows SKLearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
    """

    def __init__(self, diameter: int, seed: int, gpu: int):
        """Deep-SVDD model architecture.

        Args:
            D (int)                 : 
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.diameter = diameter
        self.net = self._build_network()
        self.rep_dim = diameter // 4

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.diameter, self.diameter // 2),
            nn.Tanh(),
            nn.Linear(self.diameter // 2, self.diameter // 4)
        )

    def forward(self, x_tensor: torch.Tensor):
        """Forward function.

        Args:
            x_tensor (torch.Tensor): Batch tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.net(x_tensor)

    def get_params(self) -> dict:
        """Get Deep-SVDD model parameters.

        Returns:
            dict: Diameternand dimensions.
        """
        return {'D': self.diameter, 'rep_dim': self.rep_dim}

def fit_with_early_stopping(train_loader, val_loader, model, patience, num_epochs, learning_rate,
                            writer, center=None, radius=0.0, objective='soft', verbose=True):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-16)

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
    radius = torch.tensor(radius, device=model.device)
    update_each = 5
    nu_val = 0.1
    # objective = 'soft'

    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            # logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            logging.debug('Epoch %d/%d.', epoch + 1, num_epochs)
            #if verbose:
                #GPUtil.showUtilization()
            # Train the model
            #logger.debug("Begin training...")
            train_loss, radius = train(train_loader, model, optimizer, epoch, center, radius,
                                objective, update_each, nu_val)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader, model, center, radius, objective, nu_val)

            if verbose:
                # logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.2f}
                # - Val loss: {val_loss:.2f}")
                logger.info("Epoch: [%d/%d] - Train loss: %2f - Val loss: %2f",
                            epoch+1, num_epochs, train_loss, val_loss)

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

    return model, center, radius

def train(train_loader, model, optimizer, epoch, center, radius, objective='soft',
        update_each=5, nu_val=0.1):
    """The training step.


    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        center (torch.Tensor)           : Hypersphere center.
        radius (float)                  : Hypersphere radius.
        objective (str, optional)       : Objective function to use. Defaults to 'soft'.
        update_each (int, optional)     : Frequency to update radius. Defaults to 5.
        nu_val (float, optional)        : Deep-SVDD hyperparameters on outliers. Defaults to 0.1.

    Returns:
        torch.Tensor                    : Training loss.
    """
    # Compute statistics
    loss_meter = AverageMeter()

    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)
        output = model(ts_batch)

        dist = torch.sum((output - center)**2, dim=1)
        if objective=='soft':
            scores = dist - radius **2
            loss = radius**2 + (1/nu_val) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Update radius
        if (objective=='soft') and (epoch >=update_each):
            radius.data = torch.tensor(get_radius(dist, nu_val), device=model.device)


        # multiplying by length of batch to correct accounting for incomplete batches
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg, radius

def validation(val_loader, model, center, radius, objective='soft', nu_val=0.1):
    """The validation step.


    Args:
        val_loader (Dataloader)         : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        center (torch.Tensor)           : Hypersphere center.
        radius (float)                  : Hypersphere radius.
        objective (str, optional)       : Objective function to use. Defaults to 'soft'.
        nu_val (float, optional)        : Deep-SVDD hyperparameters on outliers.
                                            Defaults to 0.1.

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
                scores = dist - radius **2
                loss = radius**2 + (1/nu_val) * \
                torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss_meter.update(loss.item())
        return loss_meter.avg

@torch.no_grad()
def predict_test_scores(model, test_loader, center, radius, objective):
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
        radius (float)                       : Hypersphere radius.
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
            score = dist - radius **2
        else:
            score = dist
        scores.extend(score.cpu().tolist())

    return np.array(scores)

def init_center_c(model, train_loader: DataLoader, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data.
        https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py

    Args:
        model (torch.nn): Torch model.
        train_loader (DataLoader): Train dataloader
        eps (float, optional): Epsilon value. Defaults to 0.1.

    Returns:
        torch.Tensor: Hyperspace center.
    """
    n_samples = 0
    center = torch.zeros(model.rep_dim, device=model.device)

    model.eval()
    with torch.no_grad():
        for sample in train_loader:
            # get the inputs of the batch
            x_tensor = sample.float().to(model.device)
            outputs = model(x_tensor)
            n_samples += outputs.shape[0]
            center += torch.sum(outputs, dim=0)

    center /= n_samples

    # If c_i is too close to 0, set to +-eps.
    # Reason: a zero unit can be trivially matched with zero weights.
    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps

    return center

def get_radius(dist: torch.Tensor, nu_val: float):
    """Get hypersphere radius.

    Args:
        dist (torch.Tensor): _description_
        nu_val (float): Outliers hyperparameters.

    Returns:
        np.array: Radius array.
    """
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1-nu_val)
