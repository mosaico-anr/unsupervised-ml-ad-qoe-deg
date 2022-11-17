# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from src.utils.algorithm_utils import Algorithm, PyTorchUtils
from src.utils.data_processing import get_train_data_loaders
from src.models.lstm_vae_utils import LSTM_VAE, fit_with_early_stopping, predict_test_scores



class LSTM_VAE_Algo(Algorithm, PyTorchUtils):
    def __init__(self, name: str='lstm_vae', num_epochs: int=100, batch_size: int=128, 
                lr: float=1e-3, lstm_dim : int=15, hidden_size: int=3, window_size: int=10,
                train_val_percentage: float=0.25, verbose=True, seed: int=None,
                gpu: int=None, patience: int=20, save_dir='data', multi_outputs=True):
        """LSTM-VAE algorithm for anomaly detection.

           Based on implementations from https://github.com/paya54/Anomaly_Detect_LSTM_VAE
           and https://github.com/TimyadNyda/Variational-Lstm-Autoencoder.

        Args:
            name (str, optional)                    : Algorithm's name. 
                                                      Defaults to 'lstm_vae'.
            num_epochs (int, optional)              : The number max of epochs. 
                                                      Defaults to 10.
            batch_size (int, optional)              : The batch size. Defaults to 128.
            lr (float, optional)                    : The optimizer learning rate.
                                                      Defaults to 1e-3.
            lstm_dim (int, optional)                : The LSTM hidden size. 
                                                      Defaults to 15.
            hidden_size (int, optional)             : The AE hidden size. 
                                                      Defaults to 5.
            window_size (int, optional)             : The size of the moving window. 
                                                      Defaults to 10.
            train_val_percentage (float, optional)  : The ratio val/train. 
                                                      Defaults to 0.25.
            verbose (bool, optional)                : Defaults to True.
            seed (int, optional)                    : The random generator seed. 
                                                      Defaults to None.
            gpu (int, optional)                     : The number of the GPU device to use. 
                                                      Defaults to None.
            patience (int, optional)                : The number of epochs to wait for 
                                                      early stopping. Defaults to 2.
            save_dir (str, optional)                : The folder to save the outputs. 
                                                      Defaults to 'data'.
        """
        
        Algorithm.__init__(self, __name__, name, seed)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.torch_save = True
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.lstm_dim = lstm_dim
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.verbose = verbose
        self.multi_outputs = multi_outputs
        # Get Tensorboard writer
        self.save_dir = save_dir #os.path.join(save_dir, f"{name}_outputs")
        #self.writer = SummaryWriter(self.save_dir)
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "lstm_dim" : lstm_dim,
                            "hidden_size": hidden_size,
                            "window_size": window_size,
                            "train_val_percentage": train_val_percentage,
                            "seed": seed,
                            "gpu": gpu,
                            "patience": patience,
                            "verbose" : verbose,
                            "multi_outputs": multi_outputs
                            }

        self.additional_params = dict()
        
    def anomaly_vector_construction(self, test_recons_errors, norm=2, sigma=3):
        """Apply the 3-sigma threshold to build anomaly vector prediction.

        Args:
            test_recons_errors (np.array): Reconstruction error vectors.
            norm (int, optional): The norm to used for normalization. Defaults to 2.
            sigma (int, optional): Sigma multiplicator coefficient. Defaults to 3.
            return_anomalies_score (bool, optional): If the normalized anomaly score must be returned.
                                                     Defaults to False.

        Returns:
            np.array: Anomaly vector.
        """
        # Check if mean error and std error has been compute
        try:
            train_std_err = self.additional_params['train_error_std']
            train_mean_err = self.additional_params['train_error_mean']
            
            test_norm = test_recons_errors - train_mean_err

            # if norm == 1:
            #     test_norm = np.abs(np.mean(test_norm, axis=1))
            # elif norm == 2:
            #     test_norm = np.sqrt(np.mean(test_norm**2, axis=1))
            # else:
            #     print("Norm not implemented")

            # Create anomalies vectors
            anomalies = (test_norm <= sigma*train_std_err)
            anomalies = np.array(list(map(lambda val : 0 if val else 1, anomalies)))
            return anomalies, test_norm
        except ValueError:
            print("The model has not been trained ! No train error mean or std !")
            
    def compute_additional_params(self, whole_data_loader):
        """Compute the mean and std on training data after fitting.

        Args:
            whole_data_loader (Dataloader): Training dataloader.
        """

        reconstr_errors = []
        with torch.no_grad():
            for ts_batch in whole_data_loader:
                #print(ts_batch[:,-1].shape)
                ts_batch = ts_batch.float().to(self.model.device)
                _, _, log_px, _ = self.model(ts_batch)
                #recons_loss = self.model.reconstruct_loss(ts_batch[:, -1], mu_x, sigma_x)
                #loss = recons_loss + kl_loss
                log_px = log_px.view(log_px.shape[0], -1)
                mean_log_likehood = torch.mean(log_px, dim=1)
                reconstr_errors.append(mean_log_likehood.cpu().numpy())
        #print(len(reconstr_errors))
        if len(reconstr_errors) > 0:
            reconstr_errors = np.concatenate(reconstr_errors)
        #print((reconstr_errors.shape))
        self.additional_params['train_error_mean'] = np.mean(reconstr_errors, axis=0)
        self.additional_params['train_error_std'] = np.std(reconstr_errors, axis=None)
            
        

    def fit(self, train_data : np.array, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (np.array): Training data.
            categorical_columns (list, optional): Column to be one-hot encoded.
                                                Defaults to None.
        """
        # Select columns to keep
        all_columns = np.arange(train_data.shape[-1])
        if categorical_columns is not None:
            numerical_columns = np.setdiff1d(all_columns, np.array(categorical_columns))
        else:
            numerical_columns = all_columns.copy()
       
        # Create the preprocessing steps
        numerical_processor = StandardScaler()
        if categorical_columns is not None:
            categorical_processor = OneHotEncoder(handle_unknown="ignore")
            processor = ColumnTransformer([
                ('one-hot', categorical_processor, categorical_columns),
                ('scaler', numerical_processor, numerical_columns)
            ])
        else :
            processor = ColumnTransformer([
                ('scaler', numerical_processor, numerical_columns)
            ])
        
        # Fit on the processor
        #print(train_data.shape)
        #print(categorical_columns, numerical_columns)
        if self.multi_outputs:
            train_data = train_data.reshape(-1, train_data.shape[-1])
        else:
            train_data = train_data.reshape(train_data.shape[0],-1)
            
        train_data = processor.fit_transform(train_data)
        self.additional_params['processor'] = processor

        train_loader, val_loader = get_train_data_loaders(train_data, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        
        self.model = LSTM_VAE(n_features=train_data.shape[1], hidden_size=self.hidden_size,
                              lstm_dim=self.lstm_dim, seed=self.seed, gpu=self.gpu)
        print(f"Fitting {self.name} model")
        writer = SummaryWriter(self.save_dir)

        self.model = fit_with_early_stopping(train_loader, val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, writer=writer, verbose=self.verbose)
        print("Fitting done")
        
        # Compute train mean/std error on whole training data
        self.model.eval()
        whole_data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        self.compute_additional_params(whole_data_loader)
        
    @torch.no_grad()
    def predict(self, test_data : np.array):
        """Predict on the test data.

        Args:
            test_data (np.array): Test data.

        Returns:
            np.array: Test predictions.
        """
        # Process the data
        if self.multi_outputs:
            test_data = test_data.reshape(-1, test_data.shape[-1])
        else:
            test_data = test_data.reshape(test_data.shape[0],-1)
            
        test_data = self.additional_params['processor'].transform(test_data)
        
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors = predict_test_scores(self.model, test_loader)
        
        
        predictions_dic = {'anomalies' : None,
                           'anomalies_score' : reconstr_errors
                          }
        return predictions_dic