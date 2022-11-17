import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.svm import OneClassSVM

from src.utils.algorithm_utils import Algorithm

class OC_SVM(Algorithm):
    def __init__(self, name='OC_SVM', kernel='rbf', degree=3, gamma='auto',
        coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False,
        max_iter=-1, save_dir=None, window_size: int=10, seed: int=None, multi_outputs=True):
        """One-Class SVM algorithm for anomaly detection.

        Args:
            name (str, optional)                    : Algorithm's name. Defaults to 'OC_SVM'.
            kernel (str, optional)                  : The kernel to use for the algorithm.
                                                      Defaults to 'rbf'.
            degree (int, optional)                  : The degree of polynomial kernel.
                                                      Defaults to 3.
            gamma (str, optional)                   : Kernel coefficient for kernel. 
                                                      Defaults to 'auto'.
            coef0 (float, optional)                 : Independent term in kernel. 
                                                      Defaults to 0.0.
            tol (float, optional)                   : Tolerance for stopping criterion. 
                                                      Defaults to 0.001.
            nu (float, optional)                    : The upper bound of anomalies in the dataset. 
                                                      Defaults to 0.5.
            shrinking (bool, optional)              : Whether the shrinking heuristic is used. 
                                                      Defaults to True.
            cache_size (int, optional)              : The size of the kernel cache. 
                                                      Defaults to 200.
            verbose (bool, optional)                : Defaults to False.
            max_iter (int, optional)                : The max number of iterations. 
                                                      Defaults to -1.
            save_dir ([type], optional)             : Folder to save the outputs. 
                                                      Defaults to None.
            window_size (int, optional)             : The size of the moving window. 
                                                      Defaults to 10.
            seed (int, optional)                    : Random seed. Defaults to None.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.multi_outputs = multi_outputs
        self.verbose = verbose
        self.max_iter = max_iter
        self.window_size = window_size
        #self.details = {}
        self.seed = seed
        self.init_params = {"name": name,
                            "kernel": kernel,
                            "degree": degree,
                            "gamma": gamma,
                            "coef0": coef0,
                            "tol": tol,
                            "nu": nu,
                            "shrinking": shrinking,
                            "cache_size": cache_size,
                            "verbose": verbose,
                            'multi_outputs': multi_outputs,
                            "max_iter": max_iter,
                            "window_size": window_size,
                            "seed": seed,
                            }
        self.additional_params = dict()
        
        
    def convert_predictions(self,pred):
        """Convert OC-SVM predictions from {-1;1} ({outlier;inlier}) to {0;1} ({inlier;outlier}).

        Args:
            pred : A prediction of the model. {-1;1} ({outlier;inlier})

        Returns:
                   The prediction converted. {0;1} ({inlier;outlier})
        """
        return (-pred+1.0)/2
        
    def fit(self, train_data : pd.DataFrame, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (pd.DataFrame): Training dataframe.
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
        pca = PCA(n_components=0.7)
        # pca.fit(train_data)
        
        #self.additional_params["pca_expl_var"] = pca.explained_variance_
        train_data = pca.fit_transform(train_data)
        self.additional_params["pca"] = pca
        self.additional_params['processor'] = processor
        
        self.model = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                 coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking,
                                 cache_size=self.cache_size, verbose=self.verbose, max_iter=self.max_iter)
        print("Fitting OC-SVM model")
        self.model.fit(train_data)
        print("Fitting done")

    def predict(self, test_data : pd.DataFrame):
        """Predict on the test dataframe

        Args:
            test_data (pd.DataFrame): Test dataframe.
            if_shap (bool, optional): If Shap values is computed during prediction. Defaults to True.

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

        
        # binary score
        anomalies = self.convert_predictions(self.model.predict(test_data).reshape(-1))
        
         #score_t = np.concatenate([padding, score_t])

        score = (-1.0)*self.model.decision_function(test_data)
        
        predictions_dict = {'anomalies': anomalies,
                            'anomalies_score' : score
                            }
        return predictions_dict