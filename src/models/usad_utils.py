import logging, sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from typing import Tuple, List

from src.utils.algorithm_utils import PyTorchUtils, AverageMeter

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class USADModel(nn.Module, PyTorchUtils):
    def __init__(self, input_length: int,
                 hidden_size: int, seed: int, gpu: int):
        """Auto-Encoder model architecture.

        Args:
            input_length (int)      : The number of input features.
            hidden_size (int)       : The hidden size.
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        # input_length = n_features * sequence_length

        enc_layers = [
            (input_length, input_length//2, nn.ReLU()),
            (input_length//2, input_length//4, nn.ReLU()),
            (input_length//4, hidden_size, nn.ReLU())
        ]
        dec_layers = [
            (hidden_size, input_length//4, nn.ReLU()),
            (input_length//4, input_length//2, nn.ReLU()),
            (input_length//2, input_length, nn.Sigmoid())
        ]

        self.encoder = self._make_linear(enc_layers)
        self.decoder_1 = self._make_linear(dec_layers)
        self.decoder_2 = self._make_linear(dec_layers)

    def _make_linear(self, layers: List[Tuple]):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
            if act_fn:
                net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def forward(self, ts_batch: torch.Tensor):
        z = self.encoder(ts_batch)
        w1 = self.decoder_1(z)
        w2 = self.decoder_2(z)
        w2_t = self.decoder_2(self.encoder(w1))
        return w1, w2, w2_t

    def compute_loss(self, ts_batch, w1, w2, w2_t, train_epoch):
        """Compute loss function

        Args:
            ts_batch (torch.Tensor): Batch inputs.
            w1 (torch.Tensor): Outputs from AE 1.
            w2 (torch.Tensor): Outputs from AE 2.
            w2_t (torch.Tensor): Outputs from boths AEs.
            train_epoch (int): Epoch number.

        Returns:
            torch.Tensor: Training loss.
        """
        loss_fn = torch.nn.MSELoss()
        recons_err_1 = loss_fn(w1, ts_batch)
        recons_err_2 = loss_fn(w2, ts_batch)
        advers_err = loss_fn(w2_t, ts_batch)
        loss_1 = 1/train_epoch * recons_err_1 + (1 - 1/train_epoch)*advers_err
        loss_2 = 1/train_epoch * recons_err_2 - (1 - 1/train_epoch)*advers_err
        return loss_1, loss_2

    def compute_test_score(self, ts_batch, w1, w2):
        """Compute test score.

        Args:
            ts_batch (torch.Tensor): Batch inputs.
            w1 (torch.Tensor): Outputs from AE 1.
            w2 (torch.Tensor): Outputs from AE 2.

        Returns:
            torch.Tensor: Anomaly score.
        """
        # loss_fn = torch.nn.MSELoss()
        recons_err_1 = torch.mean((ts_batch - w1)**2, axis=1) #loss_fn(w1, ts_batch)
        recons_err_2 = torch.mean((ts_batch - w2)**2, axis=1) #loss_fn(w2, ts_batch)

        return recons_err_1, recons_err_2


def fit_with_early_stopping(train_loader, val_loader, model, patience, num_epochs, lr,
                            writer, verbose=True):
    """The fitting function of the Auto Encoder.

    Args:
        train_loader (Dataloader)   : The train dataloader.
        val_loader (Dataloader)     : The val dataloader.
        model (nn.Module)           : The Pytorch model.
        patience (int)              : The number of epochs to wait for early stopping.
        num_epochs (int)            : The max number of epochs.
        lr (float)                  : The learning rate.
        writer (SummaryWriter)      : The Tensorboard Summary Writer.
        verbose (bool, optional)    : Defaults to True.

    Returns:
                        [nn.Module ]: The fitted model.
    """
    model.to(model.device)  # .double()
    optimizer_1 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder_1.parameters()), lr=lr)
    optimizer_2 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder_2.parameters()), lr=lr)
    
    model.train()
    #train_loss_by_epoch = []
    #val_loss_by_epoch = []
    best_val_loss = np.inf
    epoch_wo_improv = 0
    best_params = model.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            #if verbose:
                #GPUtil.showUtilization()
            # Train the model
            #logger.debug("Begin training...")
            train_loss_1, train_loss_2 = train(train_loader, model, optimizer_1, optimizer_2, epoch)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss_1, val_loss_2 = validation(val_loader, model, epoch)
            
            if verbose:
                logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss 1: {train_loss_1:.2f} / Train loss 2: {train_loss_2:.2f} \
                            - Val loss 1: {val_loss_1:.2f} / Val loss 2: {val_loss_2:.2f}")
            
            # Write in TensorBoard
            writer.add_scalar('train_loss_1', train_loss_1, epoch)
            writer.add_scalar('val_loss_1', val_loss_1, epoch)
            writer.add_scalar('train_loss_2', train_loss_2, epoch)
            writer.add_scalar('val_loss_2', val_loss_2, epoch)
            
            # Check if the validation loss improve or not
            if (val_loss_1 + val_loss_2 < best_val_loss)  :
                best_val_loss = val_loss_1 + val_loss_2
                epoch_wo_improv = 0
                best_params = model.state_dict()
            elif val_loss_1 + val_loss_2 >= best_val_loss:
                epoch_wo_improv += 1
            
        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break

    return model

def train(train_loader, model, optimizer_1, optimizer_2, epoch):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer_1 (torch.optim)       : The Optimizer of first AE.
        optimizer_2 (torch.optim)       : The Optimizer of second AE.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    
    #
    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)

        w1, w2, w2_t = model(ts_batch)
        loss1, loss2 = model.compute_loss(ts_batch, w1, w2, w2_t, epoch+1)
        
        loss = loss1 + loss2

        loss.backward()
        
        optimizer_1.step()
        optimizer_1.zero_grad()
        optimizer_2.step()
        optimizer_2.zero_grad()

        # multiplying by length of batch to correct accounting for incomplete batches
        loss1_meter.update(loss1.item())
        loss2_meter.update(loss2.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss1_meter.avg, loss2_meter.avg
    
def validation(val_loader, model, epoch):
    """The validation step.

    Args:
        val_loader (Dataloader)         : The val data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """

    # Compute statistics
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    
    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)

            w1, w2, w2_t = model(ts_batch)
            loss1, loss2 = model.compute_loss(ts_batch, w1, w2, w2_t, epoch+1)

            # multiplying by length of batch to correct accounting for incomplete batches
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
        return loss1_meter.avg, loss2_meter.avg
    
@torch.no_grad()
def predict_test_scores(model, test_loader, alpha=.5, beta=.5):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.
        alpha (float, optional)         : USAD trade-off parameters. Defaults to 0.5.
        beta (float, optional)          : USAD trade-off parameters. Defaults to 0.5.

    Returns:
                The reconstruction score 
    """
    model.eval()
    reconstr_scores = []

    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(model.device)

        w1 = model.decoder_1(model.encoder(ts_batch))
        w2 = model.decoder_2(model.encoder(w1))

        score1, score2 = model.compute_test_score(ts_batch, w1, w2)
        score = alpha*score1 + beta*score2
        reconstr_scores.append(score.cpu().numpy().tolist())
    reconstr_scores = np.concatenate(reconstr_scores)
    return reconstr_scores