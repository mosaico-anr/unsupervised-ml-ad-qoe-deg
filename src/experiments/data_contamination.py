"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import argparse
import os
import time
import contextlib

import numpy as np
import torch

from src.utils.algorithm_utils import save_torch_algo, load_torch_algo
from src.utils.data_processing import data_processing
from src.utils.evaluation_utils import get_best_score, get_performance
from src.models.anomaly_pca import AnomalyPCA
from src.models.oc_svm import OCSVM
from src.models.iforest import IForest
from src.models.auto_encoder import AutoEncoder
from src.models.lstm_vae import LSTMVAEAlgo
from src.models.dagmm import DAGMM
from src.models.deep_svd import DeepSVDD
from src.models.usad import USAD



all_models = ['PCA', 'IForest', 'OC-SVM', 'AE', 'Deep-SVDD', 'USAD', 'DAGMM', 'LSTM-VAE']
all_algo = [AnomalyPCA, IForest, OCSVM, AutoEncoder, DeepSVDD, USAD, DAGMM, LSTMVAEAlgo]
assert(len(all_models) == len(all_algo))
models_algo_name_map = {all_models[i]: all_algo[i] for i in range(len(all_models))}
metrics_list = ['pw', 'window_wad', 'window_pa', 'window_pak', 'window_rpa']
platforms = ['std', 'gfn', 'xc']

def contamination_experiment(model_name, save_dir, data_dir, n_runs=5, window_size=10,
                            cont_rate=0, threshold=10, verbose=False, platform='std',
                            wad_threshold=8, is_trained=False, is_consecutive=False):
    """ Launch data contamination experiment on unsupervised ML models. 

    Args:
        model_name (str)                : Model name.
        save_dir (str)                  : The path to save the experiment outputs.
        data_dir (str)                  : The path to the datasets
        n_runs (int, optional)          : The number of runs for the experiment. Defaults to 5.
        window_size (int, optional)     : The window size. Defaults to 10.
        cont_rate (int, optional)       : The data contamination rate. Defaults to 0.
        threshold (int, optional)       : The threshold to consider a window as anomalous. 
                                        Defaults to 10.
        wad_threshold (int, optional)   : WAD threshold. Defaults to 8.
        verbose (bool, optional)        : If the information should be print or not. 
                                        Defaults to False.
        platform (str, optional)        : The platform dataset to use. Defaults to std.
        is_trained (bool, optional)     : If the model is already trained. Load the model 
                                        outputs and evaluate only if true. Defaults to False.
        is_consecutive (bool, optional) : To ignore. Defaults to False.
    """
    model_save_path = f'{save_dir}/{model_name}/'
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    result_file = f'{model_save_path}/results_{platform}.txt'

    with open(result_file, 'w', encoding='utf-8') as file:
        with contextlib.redirect_stdout(file):
            performance_metrics = {'train_time' : [], 'test_time': [] }
            for met in metrics_list:
                performance_metrics[met] = { 'f1' : [],'recall' : [], 'precision' : [], 'auc' : [],
                                            'mcc' : [], 'acc_anom' : [], 'acc_norm' : [] }

            print(f'Begin experiments on {model_name} algorithm...')

            for i in range(n_runs):

                # Get the data
                data_random, categorical_ind = data_processing(data_dir,
                                                                window_size=window_size,
                                                                contamination_ratio=cont_rate,
                                                                wad_threshold=wad_threshold,
                                                                seed=42+i,
                                                                threshold=threshold,
                                                                is_consecutive=is_consecutive,
                                                                platform=platform)

                # Create the model
                model_i_path = f'{model_save_path}model_{i+1}/'
                model_alg = models_algo_name_map.get(model_name)
                is_torch_model = model_name not in ['OC-SVM', 'IForest', 'PCA']

                # Load the model if already trained
                if is_trained:
                    # Load the model
                    algo_config_filename = f'{model_i_path}init_params'
                    saved_model_filename = f'{model_i_path}model'
                    additional_params_filename = f'{model_i_path}additional_params'
                    model = load_torch_algo(model_alg, algo_config_filename, saved_model_filename,
                                            additional_params_filename,
                                            evaluation=True,
                                            torch_model=is_torch_model)
                    train_time = -1
                    performance_metrics['train_time'].append(train_time)
                else:
                    gpu_args = {"gpu":0 if torch.cuda.is_available() else None}

                    if model_name not in ['OC-SVM', 'IForest', 'PCA']:
                        model = model_alg(save_dir=model_i_path, **gpu_args, verbose=verbose,
                                        patience=10)
                    else:
                        model = model_alg(save_dir=model_i_path)

                    # Train the model
                    categorical_columns = \
                    categorical_ind if (model_name in ['OC-SVM', 'IForest']) else None

                    start = time.time()
                    model.fit(data_random['train']['data'],
                            categorical_columns=categorical_columns)
                    train_time = time.time() - start
                    performance_metrics['train_time'].append(train_time)

                    # Save the model

                    if not os.path.isdir(model_i_path):
                        os.mkdir(model_i_path)
                    save_torch_algo(model, model_i_path, torch_model=is_torch_model)

                # Evaluate on the test set
                # Evaluate on the test set
                start = time.time()
                an_dict = model.predict(data_random['test']['data'])
                test_time = time.time() - start
                performance_metrics['test_time'].append(test_time)
                model_score = an_dict['anomalies_score']

                for metric in metrics_list:
                    if metric.startswith('window'):
                        metric, window_type = metric.split('_')
                    else:
                        window_type = 'wad'

                    if model_name in ['OC-SVM', 'IForest']:

                        #score_norm = model_score
                        y_pred = an_dict['anomalies']
                        # print(y_pred[:100])
                        # print(data_random['test']['data'][0])
                        # print(data_random['test']['labels'][:10])
                        # break


                        #print(score_norm.shape, y_pred.shape)
                        precision, recall, f1_score, auc, mcc, acc_anom, acc_norm = \
                            get_performance(y_pred, y_true=data_random['test']['labels'],
                                            test_score=model_score,
                                            y_win_adjust=data_random['test']['window_adjust'],
                                            metric_type=metric,
                                            window_type=window_type)

                    elif model_name in ['PCA']:
                        # Reconstruction-based model compute the test score based
                        # on the best threshold
                        score_norm = model_score.mean(axis=1)
                        score_norm.reshape(score_norm.shape[0],-1)
                        precision, recall, f1_score, auc, mcc, acc_anom, acc_norm = \
                            get_best_score(test_score=score_norm,
                                        y_true=data_random['test']['labels'],
                                        y_win_true=data_random['test']['window_labels'],
                                        y_win_adjust=data_random['test']['window_adjust'],
                                        val_ratio=0.2, n_pertiles=100,
                                        metric_type=metric, window_type=window_type)
                    else:
                        #score_norm = model_score
                        score_norm = model_score.reshape(model_score.shape[0],-1)
                        precision, recall, f1_score, auc, mcc, acc_anom, acc_norm = \
                            get_best_score(test_score=score_norm,
                                        y_true=data_random['test']['labels'],
                                        y_win_true=data_random['test']['window_labels'],
                                        y_win_adjust=data_random['test']['window_adjust'],
                                        val_ratio=0.2, n_pertiles=100,
                                        metric_type=metric, window_type=window_type)
                    if metric == 'window':
                        metric = f'{metric}_{window_type}'
                    # Add metrics in dict
                    performance_metrics[metric]['precision'].append(precision)
                    performance_metrics[metric]['recall'].append(recall)
                    performance_metrics[metric]['f1'].append(f1_score)
                    performance_metrics[metric]['auc'].append(auc)
                    performance_metrics[metric]['mcc'].append(mcc)
                    performance_metrics[metric]['acc_anom'].append(acc_anom)
                    performance_metrics[metric]['acc_norm'].append(acc_norm)

            # Compute the average/std of the results and write in the file
            print('---' * 40)
            print('Performance results :\n')
            print(f"Training time : {np.mean(performance_metrics['train_time']):.4f} \
                    (+/-{np.std(performance_metrics['train_time']):.4f})\n")
            print(f"Testing time : {np.mean(performance_metrics['test_time']):.4f} \
                    (+/-{np.std(performance_metrics['test_time']):.4f})\n")
            inf_time_per_window = np.array(performance_metrics['test_time']) / \
                    len(data_random['test']['data'])
            print(f"Inference time per window : {np.mean(inf_time_per_window):.6f} \
                    (+/-{np.std(inf_time_per_window):.6f})")

            for metric in metrics_list:

                print(f'Evaluation with {metric} metrics:\n')
                for metric_name, metrics in performance_metrics[metric].items():
                    print(f"{metric_name} : {np.mean(metrics):.4f}(+/-{np.std(metrics):.4f})")
                print('---' * 20)

            print('Experiments finished !')

def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    # all_models = ['PCA', 'OC-SVM', 'IForest', 'AE', 'LSTM-VAE', 'DAGMM', 'Deep-SVDD', 'USAD']
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', type=int, default=10,
                        help='The window size. Default is 10.')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='The threshold of anomalous observations to consider in a window. \
                                Default is 0.8.')
    parser.add_argument('--wad-threshold', type=float, default=0.8,
                        help='WAD approach (alpha) threshold. Default is 0.8.')
    parser.add_argument('--contamination-ratio', type=float, default=0.0,
                        help='The contamination ratio. Default is 0.')
    parser.add_argument('--platform', type=str, default='std', choices=platforms,
                        help='The platform data to use. Default is std.')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='The number of time each experiment is runned. Default is 5.')
    parser.add_argument('--models-list', nargs='*', default=all_models, choices=all_models,
                        help='The list of models to evaluate.')
    parser.add_argument('--save-dir', type=str,
                        default='/data/outputs/contamination_exps',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='/data/outputs_csv',
                        help='The folder where the data are stored.')
    parser.add_argument('--is-trained', action='store_true',
                        help='If the models are already trained. Default action is false.')
    return parser.parse_args()

def run_all_experiments(args):
    """Experiments running function.

    Args:
        args (): Argparse args.

    Raises:
        ValueError: The threshold must be a float between 0 and 1.
    """
    save_dir = args.save_dir
    data_dir = args.data_dir
    n_runs = args.n_runs
    window_size = args.window_size
    threshold = args.threshold
    wad_threshold = args.wad_threshold
    cont_rate = args.contamination_ratio
    platform = args.platform
    models_list = args.models_list
    is_trained = args.is_trained
    # print(is_trained)
    # Check if the threshold is between 0 and 1
    if (threshold<=1 and threshold>=0):
        threshold_int = int(window_size*threshold)
        wad_threshold_int = int(window_size*wad_threshold)
        print(threshold_int)
    else:
        raise ValueError("The threshold must be a float between 0 and 1 !")
    # # Check if the models list is in the list of all models
    # if not set(models_list).issubset(all_models):
    #     raise ValueError(f'The list of models must be in {all_models}')

    contamination_rate = [0.0, 0.04, 0.08, 0.12, 0.2]

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    print("Start the experiments...")
    for cont_rate in contamination_rate:
        print(f"Experiments with data contamination of {cont_rate*100:.2f}%")
        cont_save_path = f"{save_dir}/cont_{cont_rate}"
        if not os.path.isdir(cont_save_path):
            os.mkdir(cont_save_path)

        print(f"Experiments with window size of {window_size}")
        win_save_path = f"{cont_save_path}/win_{window_size}"
        if not os.path.isdir(win_save_path):
            os.mkdir(win_save_path)
        # Run the experiments for all the algorithms
        for model_name in models_list:
            print(f"Experiments with {model_name} started ...")
            contamination_experiment(model_name=model_name, save_dir=win_save_path,
                                    data_dir=data_dir, n_runs=n_runs,
                                    window_size=window_size,
                                    platform=platform,
                                    cont_rate=cont_rate, threshold=threshold_int,
                                    wad_threshold=wad_threshold_int,
                                    is_trained=is_trained)
            print(f"Experiments with {model_name} finished !")

    print("All the experiments are completed !")

if __name__ == '__main__':
    arguments = parse_arguments()
    run_all_experiments(arguments)
