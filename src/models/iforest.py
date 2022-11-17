# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

from src.utils.algorithm_utils import Algorithm

class IForest(Algorithm):
    def __init__(self, name='IForest', seed: int=None, n_estimators=100, contamination='auto',
                save_dir=None, max_features=1.0, window_size=10, multi_outputs=True):
        """Isolation Forest algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'IForest'.
            seed (int, optional)            : Random seed. Defaults to None.
            n_estimators (int, optional)    : The number of base estimators. Defaults to 100.
            contamination (str, optional)   : The amount of contamination in the dataset.
                                              Defaults to 'auto'.
            save_dir ([type], optional)     : Folder to save the outputs. Defaults to None.
            max_features (float, optional)  : The numbers of samples to draw
                                              from X to train each base estimator. Defaults to 1.0.
            window_size (int, optional)     : The size of the window moving. Defaults to 10.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.window_size = window_size
        self.save_dir = save_dir
        self.additional_params = {}
        self.multi_outputs = multi_outputs
        self.init_params = {
            'contamination' : contamination,
            'n_estimators' : n_estimators,
            'max_features' : max_features,
            'window_size' : window_size,
            'save_dir' : save_dir,
            'multi_outputs': multi_outputs
        }
        
        
    def convert_predictions(self,pred):
        """Convert IF predictions from {-1;1} ({outlier;inlier}) to {0;1} ({inlier;outlier}).

        Args:
            pred : A prediction of the model. {-1;1} ({outlier;inlier})

        Returns:
                   The prediction converted. {0;1} ({inlier;outlier})
        """
        return (-pred+1.0)/2
        
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
        
        # Create the window on the data processed
        #print(type(train_data))
#         train_data_win = get_sub_seqs(train_data, window_size=self.window_size)
        
            

#         train_data_win = train_data_win.reshape(train_data_win.shape[0],-1)            
        #print(train_data.shape)
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                    max_features=self.max_features)
        print(f"Fitting {self.name} model")
        self.model.fit(train_data)
        print("Fitting done")

    def predict(self, test_data : np.array):
        """Predict on the test data

        Args:
            test_data (np.array): Test data.

        Returns:
            dict : Test predictions.
        """
        # Process the data
        if self.multi_outputs:
            test_data = test_data.reshape(-1, test_data.shape[-1])
        else:
            test_data = test_data.reshape(test_data.shape[0],-1)
            
        test_data = self.additional_params['processor'].transform(test_data)
        
        # binary classification
        anomalies = self.convert_predictions(self.model.predict(test_data).reshape(-1))
        # anomalies = np.concatenate([padding, anomalies])
        
        # binary score
        
        #print(anomalies.shape)
        #score_t = np.concatenate([padding, score_t])

        score = (-1.0)*self.model.decision_function(test_data) #https://github.com/lukasruff/Deep-SVDD/blob/master/src/isoForest.py
                                                                    # Or use score_sample() function
        predictions_dict = {'anomalies': anomalies,
                            'anomalies_score' : score
                            }
        return predictions_dict