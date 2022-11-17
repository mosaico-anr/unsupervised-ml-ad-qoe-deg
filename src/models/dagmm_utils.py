# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import logging, sys
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from src.utils.algorithm_utils import PyTorchUtils, AverageMeter



# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Adapted from https://github.com/intrudetection/robevalanodetect

class GMM(nn.Module, PyTorchUtils):
    def __init__(self, layers : List[Tuple], seed : int, gpu : int):
        """_summary_

        Args:
            layers (List[Tuple])    : The GMM layers.
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.net = self.create_network(layers)
        self.K = layers[-1][1]

    def create_network(self, layers: List[Tuple]):
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            if in_neuron and out_neuron:
                net_layers.append(nn.Linear(in_neuron, out_neuron))
            net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class AEModel(nn.Module, PyTorchUtils):
    def __init__(self, input_length: int,
                 hidden_size: int, seed: int, gpu: int):
        """Auto-Encoder model architecture.

        Args:
            input_length (int)          : The number of input features.
            hidden_size (int)           : The hidden size.
            seed (int)                  : The random generator seed.
            gpu (int)                   : The number of the GPU device.
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        #input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        enc_layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        dec_layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*enc_layers)
        self.to_device(self._encoder)
        self._decoder = nn.Sequential(*dec_layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        """Forward function of the Auto-Encoder.

        Args:
            ts_batch        : The batch input.
            return_latent   : If the latent vector must be returned. 
                                Defaults to False.

        Returns:
                The reconstructed batch.
        """
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence


class DAGMMModel(nn.Module, PyTorchUtils):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, input_length: int, hidden_size: int,
                 lambda_2=0.005, lambda_1=0.1, reg_covar=1e-12,
                K: int=4,seed: int=None, gpu: int=None):
        """DAGMM model as implemented in https://github.com/danieltan07/dagmm.

        Args:
            input_length (int)              : The number of input features.
            hidden_size (int)               : The hidden size.
            lambda_2 (float, optional)      : DAGMM meta-parameters. Defaults to 0.005.
            lambda_1 (float, optional)      : DAGMM meta-parameters.. Defaults to 0.1.
            reg_covar (_type_, optional)    : Regularization parameter for PD matrix. Defaults to 1e-12.
            K (int, optional)               : _description_. Defaults to 4.
            seed (int, optional)            : The random generator seed. Defaults to None.
            gpu (int, optional)             : The number of the GPU device. Defaults to None.
        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.ae = None
        self.gmm = None
        self.K = K
        self.seed = seed
        self.gpu = gpu
        # self.name = "DAGMM"
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)
        self.resolve_params()

    def resolve_params(self):
        # defaults to parameters described in section 4.3 of the paper
        # https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
        gmm_layers = [
            (self.hidden_size + 2, 10, nn.Tanh()),
            (10, 20, nn.Tanh()),
            (None, None, nn.Dropout(0.5)),
            (20, self.K, nn.Softmax(dim=1))
        ]
        self.ae = AEModel(input_length=self.input_length, hidden_size=self.hidden_size,
                          seed=self.seed, gpu=self.gpu)
        self.gmm = GMM(gmm_layers, seed=self.seed, gpu=self.gpu)

    def forward(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae._encoder(x)
        x_prime = self.ae._decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)
        # gamma = self.softmax(output)

        return code, x_prime, cosim, z_r, gamma_hat

    def forward_end_dec(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae._encoder(x)
        x_prime = self.ae._decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        return code, x_prime, cosim, z_r

    def forward_estimation_net(self, z_r: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param z_r: input
        :return: output of the model
        """

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)

        return gamma_hat

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`
        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`
        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)
        Returns
        -------
        """
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        # self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        # self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = torch.sqrt(gamma.unsqueeze(-1)) * (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov_mat = torch.sum(z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        # self.cov = cov.data
#         N = z.shape[0]
#         K = gamma.shape[1]

#         # K
#         gamma_sum = torch.sum(gamma, dim=0)
#         phi = gamma_sum / N

#         # phi = torch.mean(gamma, dim=0)

#         # K x D
#         # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
#         mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
#         # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

#         mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
#         cov_mat = mu_z.unsqueeze(-1) * mu_z.unsqueeze(-2)
#         cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
#         cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)


        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        """Sample energy computation.

        Args:
            z (torch.Tensor)                    : _description_
            phi (torch.Tensor, optional)        : Mixture-component distribution. Defaults to None.
            mu (torch.Tensor, optional)         : Mixture mean. Defaults to None.
            cov_mat (torch.Tensor, optional)    : Mixture covariance. Defaults to None.
            average_energy (bool, optional)     : if energy should be average. Defaults to True.
        """
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat
            
#         k, D, _ = cov_mat.size()

#         z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

#         cov_inverse = []
#         det_cov = []
#         cov_diag = 0
#         eps = 1e-12
#         for i in range(k):
#             # K x D x D
#             cov_k = cov_mat[i] + (torch.eye(D)).to(self.device) *self.reg_covar
#             cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
#             #det_cov.append((torch.linalg.cholesky(cov_k * 2*np.pi).diag().prod()).unsqueeze(0))
#             det_cov.append((torch.linalg.det(cov_k* 2*np.pi).cpu().detach().numpy()))
#             #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
#             cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

#         # K x D x D
#         cov_inverse = torch.cat(cov_inverse, dim=0)
#         # K
#         det_cov = torch.from_numpy(np.float32(np.array(det_cov)))
        
#         # N x K
#         exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
#         # for stability (logsumexp)
#         max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

#         exp_term = torch.exp(exp_term_tmp - max_val)

#         sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        
#         if average_energy:
#             sample_energy = torch.mean(sample_energy)

#         return sample_energy, cov_diag
            
        # jc_res = self.estimate_sample_energy_js(z, phi, mu)

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        #eps = self.reg_covar
        eps = torch.abs(torch.min(torch.real(torch.linalg.eigvals(cov_mat)))) + 1e-4
        # eps = 1e-6
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        try:
            inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
            det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        except torch.linalg.LinAlgError as e:
            print(e)
            print(cov_mat.shape)
            print(torch.linalg.eigvals(cov_mat), torch.linalg.eigvals(cov_mat).shape)
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        
        # inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        

        
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_prime, energy, pen_cov_mat):
        """Loss computation

        Args:
            x (torch.Tensor)            : Input tensor.
            x_prime (torch.Tensor)      : Reconstructed tensor.
            energy (torch.Tensor)       : Sample energy.
            pen_cov_mat (torch.Tensor)  : Penalization.

        Returns:
            torch.Tensor: Batch loss.
        """
        rec_err = ((x - x_prime) ** 2).mean()
        loss = rec_err + self.lambda_1 * energy + self.lambda_2 * pen_cov_mat

        return loss

    def get_params(self) -> dict:
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "latent_dim": self.ae.latent_dim,
            "K": self.gmm.K
        }

def fit_with_early_stopping(train_loader, val_loader, model, patience, num_epochs, lr,
                            writer, verbose=True):
    """The fitting function of the DAGMM model.

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
            train_loss = train(train_loader, model, optimizer, epoch)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader, model, optimizer, epoch)
            
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
            
    return model


def train(train_loader, model, optimizer, epoch):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    loss_meter = AverageMeter()
    
    #
    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)

        # Forward pass
        code, x_prime, _, z_r, gamma_hat = model(ts_batch)
        phi, mu, cov_mat = model.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = model.estimate_sample_energy(z_r, phi, mu, cov_mat)


        loss = model.compute_loss(ts_batch, x_prime, energy_result, pen_cov_mat)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        # multiplying by length of batch to correct accounting for incomplete batches
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg
    
def validation(val_loader, model, optimizer, epoch):
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
    loss_meter = AverageMeter()
    
    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)

            # Forward pass
            code, x_prime, cosim, z_r, gamma_hat = model(ts_batch)
            phi, mu, cov_mat = model.compute_params(z_r, gamma_hat)
            energy_result, pen_cov_mat = model.estimate_sample_energy(z_r, phi, mu, cov_mat)
            
            loss = model.compute_loss(ts_batch, x_prime, energy_result, pen_cov_mat)
            #val_loss.append(loss.item()*len(ts_batch))
            loss_meter.update(loss.item())
        return loss_meter.avg
    
@torch.no_grad()
def predict_test_scores(model, test_loader, train_phi, train_mu, train_cov_mat):
    """The prediction step.                

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.
        train_phi (torch.Tensor)        : Training distribution matrixes.
        train_mu (torch.Tensor)         : Training mean matrixes.
        train_cov_mat (torch.Tensor)    : Training covariance matrixes.

    Returns:
        _type_: The reconstruction score.
    """
    model.eval()
    test_energy = []
    test_z = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(model.device)
        code, x_prime, cosim, z_r, gamma_hat = model(ts_batch)
        energy_result, pen_cov_mat = model.estimate_sample_energy(z_r, train_phi, train_mu, train_cov_mat, average_energy=False)
        test_energy.append(energy_result.cpu().numpy())
        test_z.append(z_r.cpu().numpy())

    test_energy = np.concatenate(test_energy, axis=0)
    test_z = np.concatenate(test_z, axis=0)

    return test_energy, test_z
