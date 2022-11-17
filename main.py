# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import argparse, time
import os

import torch

from src.models.anomaly_pca import Anomaly_PCA
from src.models.oc_svm import OC_SVM
from src.models.iforest import IForest
from src.models.auto_encoder import AutoEncoder
from src.models.lstm_vae import LSTM_VAE_Algo
from src.models.dagmm import DAGMM
from src.models.deep_svd import DeepSVDD
from src.models.usad import USAD
from src.utils.algorithm_utils import save_torch_algo, load_torch_algo
from src.utils.data_processing import data_processing
from src.utils.evaluation_utils import get_best_score, get_performance

all_models = ['PCA', 'OC-SVM', 'IForest', 'AE', 'LSTM-VAE', 'DAGMM', 'Deep-SVDD', 'USAD']
all_algo = [Anomaly_PCA, OC_SVM, IForest, AutoEncoder, LSTM_VAE_Algo, DAGMM, DeepSVDD, USAD]
assert(len(all_models) == len(all_algo))
models_algo_name_map = {all_models[i]: all_algo[i] for i in range(len(all_models))}
metrics_list = ['pw', 'window_wad', 'window_pa', 'window_rpa']

def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, choices=all_models,
                        required=True, help='The model to train and test.')
    parser.add_argument('--window-size', type=int, default=10, 
                        help='The window size. Default is 10.')
    parser.add_argument('--contamination-ratio', type=float, default=0.0, 
                        help='The contamination ratio. Default is 0.')
    parser.add_argument('--seed', type=int, default=42, 
                        help='The random generator seed. Default is 42.')
    parser.add_argument('--model-save-path', type=str, default='data/outputs/',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='data/outputs_csv/',
                        help='The folder where the data are stored.')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='The threshold of anomalous observations to consider in a window. Default is 0.8.')
    parser.add_argument('--metric', type=str, default='window_wad', choices=metrics_list,
                        required=True, help='The metric to use for evaluation.')
    parser.add_argument('--is-trained', action='store_true',
                        help='If the models are already trained. Default action is false.')

    return parser.parse_args()
            
def main(args):
    """Main function

    Args:
        args : Command-line arguments.
    """
    model_name = args.model_name
    contamination_ratio = args.contamination_ratio
    window_size = args.window_size
    seed = args.seed
    data_dir = args.data_dir
    window_size = args.window_size
    threshold = args.threshold
    is_trained = args.is_trained
    model_save_path = args.model_save_path
    metric = args.metric
    verbose = True

    # Check if the threshold is between 0 and 1
    if (threshold<=1 and threshold>=0):
        threshold_int = int(window_size*threshold)
        print(threshold_int)
    else:
        raise ValueError("The threshold must be a float between 0 and 1 !")

    if model_name not in all_models:
        raise ValueError(f"This model : {model_name} is not implemented.\n Must be in {all_models}.")
    
    print('Data loading...')
    # Get the data
    data_random, categorical_ind = data_processing(data_dir, window_size=window_size, contamination_ratio=contamination_ratio,
                                                    seed=seed, threshold=threshold_int, is_consecutive=False)
    model_alg = models_algo_name_map.get(model_name)
    is_torch_model = model_name not in ['OC-SVM', 'IForest', 'PCA']

    # Load the model if already trained
    if is_trained:
        # Load the model
        algo_config_filename = f'{model_save_path}init_params'
        saved_model_filename = f'{model_save_path}model'
        additional_params_filename = f'{model_save_path}additional_params'
        model = load_torch_algo(model_alg, algo_config_filename, saved_model_filename,
                                    additional_params_filename, eval=True, torch_model=is_torch_model)
    else:
        gpu_args = {"gpu":0 if torch.cuda.is_available() else None}

        if model_name not in ['OC-SVM', 'IForest', 'PCA']:
            model = model_alg(save_dir=model_save_path, **gpu_args, verbose=verbose, patience=10)
        else:
            model = model_alg(save_dir=model_save_path)

        # Train the model
        categorical_columns = categorical_ind if (model_name in ['OC-SVM', 'IForest']) else None

        start = time.time()
        model.fit(data_random['train']['data'], categorical_columns=categorical_columns)
        train_time = time.time() - start

        # Save the model
        if not (os.path.isdir(model_save_path)):
            os.mkdir(model_save_path)
        save_torch_algo(model, model_save_path, torch_model=is_torch_model)

    # Evaluate on the test set
    # Evaluate on the test set
    start = time.time()
    an_dict = model.predict(data_random['test']['data'])
    test_time = time.time() - start
    model_score = an_dict['anomalies_score']

    if metric.startswith('window'):
        metric, window_type = metric.split('_')
    else:
        window_type = 'wad'
    
    if model_name in ['OC-SVM', 'IForest']:
        
        #score_norm = model_score
        y_pred = an_dict['anomalies']


        #print(score_norm.shape, y_pred.shape)
        precision, recall, f1, auc, mcc, acc_anom, acc_norm = get_performance(y_pred, y_true=data_random['test']['labels'],
                                                    test_score=model_score, y_win_adjust=data_random['test']['window_adjust'],
                                                    metric_type=metric, window_type=window_type)
        
    elif model_name in ['PCA']:
        # Reconstruction-based model compute the test score based on the best threshold                     
        score_norm = model_score.mean(axis=1)
        score_norm.reshape(score_norm.shape[0],-1)
        precision, recall, f1, auc, mcc, acc_anom, acc_norm = get_best_score(test_score=score_norm, y_true=data_random['test']['labels'],
                                    y_win_true=data_random['test']['window_labels'],
                                    y_win_adjust=data_random['test']['window_adjust'],
                                    val_ratio=0.2, n_pertiles=100, metric_type=metric, window_type=window_type)
    else:
        #score_norm = model_score
        score_norm = model_score.reshape(model_score.shape[0],-1)
        precision, recall, f1, auc, mcc, acc_anom, acc_norm = get_best_score(test_score=score_norm, y_true=data_random['test']['labels'],
                                    y_win_true=data_random['test']['window_labels'],
                                    y_win_adjust=data_random['test']['window_adjust'],
                                    val_ratio=0.2, n_pertiles=100, metric_type=metric, window_type=window_type)
    if metric == 'window':
        metric = f'{metric}_{window_type}'

    ## Print statistics report
    print('---' * 40)
    print(f'Performance results for {model_name} model:\n')
    print(f"Training time : {train_time:.4f}")
    print(f"Test time : {test_time:.4f}")
    inf_time_per_window = test_time/len(data_random['test']['data'])
    print(f"Inference time per window : {inf_time_per_window:.6f})")

    print(f"Results for {metric} metric")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"MCC : {mcc:.4f}")
    print(f"AUC : {auc:.4f}")
    print(f"Accuracy normal : {acc_norm:.4f}")
    print(f"Accuracy anomalies : {acc_anom:.4f}")

    print('Experiments finished !')



    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)