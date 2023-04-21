"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from src.utils.algorithm_utils import Algorithm, PyTorchUtils
from src.utils.data_processing import get_train_data_loaders
from src.models.dagmm_utils import DAGMMModel, fit_with_early_stopping, predict_test_scores



class DAGMM(Algorithm, PyTorchUtils):
    """DAGMM algorithm.
    """
    def __init__(self, name: str='dagmm', num_epochs: int=100, batch_size: int=128,
                learning_rate: float=1e-3, hidden_size: int=5, window_size: int=10,
                lambda_2=0.005, lambda_1=0.1, reg_covar=1e-12, K: int=4,
                train_val_percentage: float=0.25, verbose=True, seed: int=None,
                gpu: int=None, patience: int=20, save_dir='data', multi_outputs=True):
        """DAGMM algorithm for anomaly detection.

        Args:
            name (str, optional)                    : Algorithm's name. 
                                                        Defaults to 'auto_encoder'.
            num_epochs (int, optional)              : The number max of epochs. 
                                                        Defaults to 10.
            batch_size (int, optional)              : The batch size. Defaults to 128.
            lr (float, optional)                    : The optimizer learning rate. 
                                                        Defaults to 1e-3.
            hidden_size (int, optional)             : The AE hidden size. 
                                                        Defaults to 5.
            window_size (int, optional)             : The size of the moving window. 
                                                        Defaults to 10.
            lambda_1 (float, optional)              : DAGMM meta-parameter.
                                                        Defaults to 0.1.
            lambda_2 (float, optional)              : DAGMM meta-parameter.
                                                        Defaults to 0.005.
            reg_covar (float, optional)             : DAGMM regularization parameter.
                                                        Defaults to 1e-12.
            K (int, optional)                       : DAGMM layers parameter.
                                                        Defaults to 4.
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
        self.learning_rate = learning_rate
        self.patience = patience

        self.hidden_size = hidden_size
        self.window_size = window_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.K = K

        self.train_val_percentage = train_val_percentage
        self.model = None
        self.verbose = verbose
        self.multi_outputs = multi_outputs
        # Get Tensorboard writer
        self.save_dir = save_dir #os.path.join(save_dir, f"{name}_opts")
        #self.writer = SummaryWriter(self.save_dir)
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "hidden_size": hidden_size,
                            "window_size": window_size,
                            "lambda_1": lambda_1,
                            "lambda_2": lambda_2,
                            "reg_covar": reg_covar,
                            "K": K,
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
            return_anomalies_score (bool, optional): If the normalized anomaly score
                                                    must be returned.
                                                    Defaults to False.

        Returns:
            np.array: Anomaly vector.
        """
        # Check if mean error and std error has been compute
        try:
            train_std_err = self.additional_params['train_error_std']
            train_mean_err = self.additional_params['train_error_mean']
            test_norm = test_recons_errors - train_mean_err

            if norm == 1:
                test_norm = np.abs(np.mean(test_norm, axis=1))
            elif norm == 2:
                test_norm = np.sqrt(np.mean(test_norm**2, axis=1))
            else:
                print("Norm not implemented")

            # Create anomalies vectors
            anomalies = test_norm <= sigma*train_std_err
            anomalies = np.array(list(map(lambda val : 0 if val else 1, anomalies)))
            return anomalies, test_norm
        except ValueError:
            print("The model has not been trained ! No train error mean or std !")


    def compute_additional_parameters(self, whole_data_loader):
        """Compute DAGMM additional parameters on training data after fitting.

        Args:
            whole_data_loader (Dataloader): Training dataloader.
        """

        N, gamma_sum, mu_sum, cov_mat_sum = 0, 0, 0, 0

        self.model.eval()
        with torch.no_grad():
            for ts_batch in whole_data_loader:
                ts_batch = ts_batch.float().to(self.model.device)

                # Forward pass
                _, _, _, z_r, gamma_hat = self.model(ts_batch)
                phi, mu_tensor, cov_mat = self.model.compute_params(z_r, gamma_hat)
                _, _ = self.model.estimate_sample_energy(z_r, phi, mu_tensor, cov_mat)
                batch_gamma_sum = gamma_hat.sum(axis=0)

                gamma_sum += batch_gamma_sum
                # keep sums of the numerator only
                mu_sum += mu_tensor * batch_gamma_sum.unsqueeze(-1)
                # keep sums of the numerator only
                cov_mat_sum += cov_mat * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
                N += ts_batch.shape[0]

            train_phi = gamma_sum/N
            train_mu = mu_sum/gamma_sum.unsqueeze(-1)
            train_cov = cov_mat_sum/gamma_sum.unsqueeze(-1).unsqueeze(-1)

            train_energy = []
            # train_labels = []
            train_z = []
            for ts_batch in whole_data_loader:
                ts_batch = ts_batch.float().to(self.model.device)

                _, _, _, z_r, gamma_hat = self.model(ts_batch)
                phi, mu_tensor, cov_mat = self.model.compute_params(z_r, gamma_hat)
                sample_energy, _ = self.model.estimate_sample_energy(z_r, phi=train_phi,
                                                                            mu=train_mu,
                                                                            cov_mat=train_cov,
                                                                            average_energy=False)

                train_energy.append(sample_energy.data.cpu().numpy())
                train_z.append(z_r.data.cpu().numpy())
                # train_labels.append(labels.numpy())


            train_energy = np.concatenate(train_energy,axis=0)
            train_z = np.concatenate(train_z,axis=0)
            # train_labels = np.concatenate(train_labels,axis=0)
            self.additional_params['train_phi'] = train_phi
            self.additional_params['train_mu'] = train_mu
            self.additional_params['train_cov_mat'] = train_cov

            self.additional_params['train_energy'] = train_energy
            self.additional_params['train_z'] = train_z
            self.additional_params['train_error_mean'] = np.mean(train_energy, axis=0)
            self.additional_params['train_error_std'] = np.std(train_energy, axis=None)


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

        # Use PCA to decorrelate data
        pca = PCA(n_components=0.9)
        # pca.fit(train_data)

        #self.additional_params["pca_expl_var"] = pca.explained_variance_
        train_data = pca.fit_transform(train_data)
        self.additional_params["pca"] = pca
        self.additional_params['processor'] = processor

        train_loader, val_loader = get_train_data_loaders(train_data,
                                                        batch_size=self.batch_size,
                                                        splits=[1 - self.train_val_percentage,
                                                        self.train_val_percentage],
                                                        seed=self.seed)

        self.model = DAGMMModel(input_length=train_data.shape[-1], hidden_size=self.hidden_size,
                                lambda_1=self.lambda_1, lambda_2=self.lambda_2,
                                reg_covar=self.reg_covar,
                                K=self.K, seed=self.seed, gpu=self.gpu)
        print(f"Fitting {self.name} model")
        writer = SummaryWriter(self.save_dir)
        self.model = fit_with_early_stopping(train_loader, val_loader, self.model,
                                            patience=self.patience,
                                            num_epochs=self.num_epochs,
                                            learning_rate=self.learning_rate,
                                            writer=writer,
                                            verbose=self.verbose)
        print("Fitting done")

        print("Compute additional parameters")

        # Compute train mean/std error on whole training data
        self.model.eval()
        whole_data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,
                                    drop_last=False,
                                    pin_memory=True,
                                    shuffle=False)
        self.compute_additional_parameters(whole_data_loader)
        print("Computing done !")

    @torch.no_grad()
    def predict(self, test_data : np.array):
        """Predict on the test data

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
        test_data = self.additional_params["pca"].transform(test_data)

        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, drop_last=False,
                                pin_memory=True,
                                shuffle=False)
        test_energy, _ = predict_test_scores(self.model, test_loader,
                                            train_phi=self.additional_params['train_phi'],
                                            train_mu=self.additional_params['train_mu'],
                                            train_cov_mat=self.additional_params['train_cov_mat'])
        predictions_dic = {'anomalies' : None,
                            'anomalies_score' : test_energy
                            }
        return predictions_dic
