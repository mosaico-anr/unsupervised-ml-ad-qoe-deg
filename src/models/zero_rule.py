"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np

from src.utils.algorithm_utils import Algorithm

class ZeroR(Algorithm):
    """Zero-Rule Algorithm."""
    def __init__(self, name='zeroR', seed: int=None, save_dir=None, multi_outputs=True):
        """Zero-Rule algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'zeroR'.
            seed (int, optional)            : Random seed. Defaults to None.
            save_dir ([type], optional)     : Folder to save the outputs. Defaults to None.

        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.seed = seed
        self.save_dir = save_dir
        self.additional_params = {}
        self.multi_outputs = multi_outputs
        self.init_params = {
            'save_dir' : save_dir,
            'seed' : seed,
            'multi_outputs': multi_outputs
        }

    def fit(self, train_data : np.array, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (np.array): Training data.
            categorical_columns (list, optional): Column to be one-hot encoded.
                                                Defaults to None.
        """


    def predict(self, test_data : np.array):
        """Predict on the test data

        Args:
            test_data (np.array): Test data.

        Returns:
            np.array: Test predictions.
        """
        np.random.seed(self.seed)
        preds = []
        pred_size = test_data.shape[1]
        for _ in range(test_data.shape[0]):
            preds.append(np.ones(pred_size,))
        anomalies = np.concatenate(preds)
        predictions_dict = {'anomalies': anomalies,
                                'anomalies_score' : anomalies
                            }
        return predictions_dict
