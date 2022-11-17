import numpy as np

from src.utils.algorithm_utils import Algorithm

class BetaDetector(Algorithm):
    def __init__(self, name='BetaDetector', seed: int=None, beta=1.0, save_dir=None, multi_outputs=True):
        """$(1-\beta)$-Detector algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'IForest'.
            seed (int, optional)            : Random seed. Defaults to None.
            save_dir ([type], optional)     : Folder to save the outputs. Defaults to None.

        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.seed = seed
        self.beta = beta
        self.save_dir = save_dir
        self.additional_params = {}
        self.multi_outputs = multi_outputs
        self.init_params = {
            'save_dir' : save_dir,
            'seed' : seed,
            'beta' : beta,
            'multi_outputs': multi_outputs
        }
        
        

        
    def fit(self, train_data : np.array, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (np.array): Training data.
            categorical_columns (list, optional): Column to be one-hot encoded.
                                                Defaults to None.
        """
        

    def predict(self, test_labels : np.array):
        """Predict on the test data

        Args:
            test_data (np.array): Test data.

        Returns:
            np.array: Test predictions.
        """
        # Test if beta is between 0 and 1
        if self.beta>1 or self.beta<0:
            raise ValueError('The value of beta must be between 0 and 1 !')
        elif self.beta == 0: # Perfect predictor
            preds = []
            pred_size = test_labels.shape[1]
            for i in range(test_labels.shape[0]):
                pred = test_labels[i].copy()
                preds.append(pred)
            anomalies = np.concatenate(preds)
            predictions_dict = {'anomalies': anomalies,
                                    'anomalies_score' : anomalies
                                   }
            return predictions_dict
        elif self.beta == 1: # Bad perfect
            preds = []
            pred_size = test_labels.shape[1]
            for i in range(test_labels.shape[0]):
                pred = 1 - test_labels[i].copy()
                preds.append(pred)
            anomalies = np.concatenate(preds)
            predictions_dict = {'anomalies': anomalies,
                                    'anomalies_score' : anomalies
                                   }
            return predictions_dict
            
        else:
            preds = []
            n_win = test_labels.shape[0]
            pred_size = test_labels.shape[1]
            # Modify beta% of the datasets
            test_labels = test_labels.copy()
            test_labels = test_labels.reshape(n_win*pred_size,)
            shuffled_indices = np.random.RandomState(seed=self.seed).permutation(pred_size*n_win)
            flipped_size = int(pred_size*n_win * self.beta)
            flipped_indices = shuffled_indices[:flipped_size]
            #print(n_win, pred_size, flipped_size, len(flipped_indices))
            test_labels[flipped_indices] = 1 - test_labels[flipped_indices]
            test_labels = test_labels.reshape(n_win, pred_size)
            for i in range(n_win):
                pred = test_labels[i].copy()
                preds.append(pred)


            anomalies = np.concatenate(preds)
            predictions_dict = {'anomalies': anomalies,
                                    'anomalies_score' : anomalies
                                   }
            return predictions_dict