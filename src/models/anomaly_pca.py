"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils.algorithm_utils import Algorithm


class AnomalyPCA(Algorithm):
    """PCA Algorithm.
    """
    def __init__(self, name='PCA', seed: int=None, n_components=0.9,
                save_dir=None,
                multi_outputs=True):
        """Anomaly PCA reconstruction algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'PCA'.
            seed (int, optional)            : Random seed. Defaults to None.
            n_components (float, optional)  : Number of principal components to keep.
                                                Defaults to 0.9.
            save_dir (str, optional)        : Folder to save the outputs.
                                                Defaults to None.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.n_components = n_components
        self.multi_outputs = multi_outputs
        self.init_params = {'n_components': n_components,
                            'save_dir' : save_dir,
                            'multi_outputs': multi_outputs
                            }
        self.additional_params = {}
        self.save_dir = save_dir


    def fit(self, train_data: np.array, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (np.array): Training dataframe.
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
            processor = numerical_processor

        # Fit the model and get error reconstruction on the training datasets
        self.model = Pipeline([('processor', processor),
                            ('pca', PCA(n_components=self.n_components))
                            ])
        print("Fitting PCA model")
        #train_data = train_data.values
        if self.multi_outputs:
            train_data = train_data.reshape(-1, train_data.shape[-1])
        else:
            train_data = train_data.reshape(train_data.shape[0],-1)
        self.model.fit(X=train_data)
        print("Fitting done !")

        #recons_train = np.dot(self.model.transform(train_data), self.model['pca'].components_)
        # + self.model['pca'].mean_
        #recons_train = self.model['scaler'].inverse_transform(recons_train)
        recons_train = self.model.inverse_transform(self.model.transform(train_data))

        # Get the reconstruction error on the training datasets
        recons_error = (train_data - recons_train)**2

        # Save min and max error for normalization of test errors.
        self.additional_params['train_error_mean'] = np.mean(recons_error, axis=0)
        self.additional_params['train_error_std'] = np.std(recons_error, axis=None)


    def anomaly_vector_construction(self, test_recons_errors, norm=2, sigma=3,
                                    return_anomalies_score=False):
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
            return anomalies, test_norm if return_anomalies_score else anomalies
        except ValueError:
            print("The model has not been trained ! No train error mean or std !")


    def predict(self, test_data: np.array):
        """Predict on the test dataframe

        Args:
            test_data (np.array): Test dataframe.

        Returns:
            np.array: Test predictions.
        """
        if self.multi_outputs:
            test_data = test_data.reshape(-1, test_data.shape[-1])
        else:
            test_data = test_data.reshape(test_data.shape[0],-1)

        recons = self.model.inverse_transform(self.model.transform(test_data))

        # Compute the reconstruction error
        if isinstance(test_data, pd.DataFrame):
            recons_error = (test_data - recons) ** 2
        else:
            recons_error = (test_data - recons) ** 2

        predictions_dict = {'anomalies_score' :  recons_error,#.mean(axis=1),
                            'recons_vect' : None,
                            'anomalies' : None,
                            #'anomalies_score' : None
        }

        return predictions_dict
