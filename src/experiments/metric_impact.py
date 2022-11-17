import argparse, os, time
import contextlib

import numpy as np
import torch

from src.utils.algorithm_utils import save_torch_algo, load_torch_algo
from src.utils.data_processing import data_processing
from src.utils.evaluation_utils import get_best_score, get_performance
from src.models.anomaly_pca import Anomaly_PCA
from src.models.oc_svm import OC_SVM
from src.models.iforest import IForest
from src.models.auto_encoder import AutoEncoder
from src.models.lstm_vae import LSTM_VAE_Algo
from src.models.dagmm import DAGMM
from src.models.deep_svd import DeepSVDD
from src.models.usad import USAD



all_models = ['PCA', 'OC-SVM', 'IForest', 'AE', 'LSTM-VAE', 'DAGMM', 'Deep-SVDD', 'USAD']
all_algo = [Anomaly_PCA, OC_SVM, IForest, AutoEncoder, LSTM_VAE_Algo, DAGMM, DeepSVDD, USAD]
assert(len(all_models) == len(all_algo))
models_algo_name_map = {all_models[i]: all_algo[i] for i in range(len(all_models))}
metrics_list = ['pw', 'window_wad', 'window_pa', 'window_rpa']

def metric_experiment(model_name, save_dir, data_dir, n_runs=5, window_size=10, cont_rate=0, threshold=10, verbose=False, is_trained=False, is_consecutive=False):
    """ Launch metric experiment on unsupervised ML models. 

    Args:
        model_name (str)                : Model name.
        save_dir (str)                  : The path to save the experiment outputs.
        data_dir (str)                  : The path to the datasets
        n_runs (int, optional)          : The number of runs for the experiment. Defaults to 5.
        window_size (int, optional)     : The window size. Defaults to 10.
        cont_rate (int, optional)       : The data contamination rate. Defaults to 0.
        threshold (int, optional)       : The threshold to consider a window as anomalous. Defaults to 10.
        verbose (bool, optional)        : If the information should be print or not. Defaults to False.
        is_trained (bool, optional)     : If the model is already trained. Load the model outputs and evaluate only if true. Defaults to False.
        is_consecutive (bool, optional) : To ignore. Defaults to False.
    """

    
    model_save_path = f'{save_dir}/{model_name}/'
    if not (os.path.isdir(model_save_path)):
        os.mkdir(model_save_path)

    result_file = f'{model_save_path}/results.txt'

    with open(result_file, 'w') as f:
        with contextlib.redirect_stdout(f):

            pw_metrics = {
                    'f1' : [],
                    'recall' : [],
                    'precision' : [],
                    'auc' : [],
                }
            performance_metrics = {'train_time' : [], 'test_time': [] }
            for met in metrics_list:
                performance_metrics[met] = { 'f1' : [],'recall' : [], 'precision' : [], 'auc' : [],
                                             'mcc' : [], 'acc_anom' : [], 'acc_norm' : [] }
            
            print(f'Begin experiments on {model_name} algorithm...')

            for i in range(n_runs):

                # Get the data
                data_random, categorical_ind = data_processing(data_dir, window_size=window_size, contamination_ratio=cont_rate,
                                                             seed=42+i, threshold=threshold, is_consecutive=is_consecutive)

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
                                               additional_params_filename, eval=True, torch_model=is_torch_model)
                    train_time = -1
                    performance_metrics['train_time'].append(train_time)
                else:
                    gpu_args = {"gpu":0 if torch.cuda.is_available() else None}

                    if model_name not in ['OC-SVM', 'IForest', 'PCA']:
                        model = model_alg(save_dir=model_i_path, **gpu_args, verbose=verbose, patience=10)
                    else:
                        model = model_alg(save_dir=model_i_path)

                    # Train the model
                    categorical_columns = categorical_ind if (model_name in ['OC-SVM', 'IForest']) else None

                    start = time.time()
                    model.fit(data_random['train']['data'], categorical_columns=categorical_columns)
                    train_time = time.time() - start
                    performance_metrics['train_time'].append(train_time)

                    # Save the model


                    if not (os.path.isdir(model_i_path)):
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
                    # Add metrics in dict
                    performance_metrics[metric]['precision'].append(precision)
                    performance_metrics[metric]['recall'].append(recall)
                    performance_metrics[metric]['f1'].append(f1)
                    performance_metrics[metric]['auc'].append(auc)
                    performance_metrics[metric]['mcc'].append(mcc)
                    performance_metrics[metric]['acc_anom'].append(acc_anom)
                    performance_metrics[metric]['acc_norm'].append(acc_norm)

            # Compute the average/std of the results and write in the file
            print('---' * 40)
            print('Performance results :\n')
            print(f"Training time : {np.mean(performance_metrics['train_time']):.4f}(+/-{np.std(performance_metrics['train_time']):.4f})\n")
            print(f"Testing time : {np.mean(performance_metrics['test_time']):.4f}(+/-{np.std(performance_metrics['test_time']):.4f})\n")
            inf_time_per_window = np.array(performance_metrics['test_time'])/len(data_random['test']['data'])
            print(f"Inference time per window : {np.mean(inf_time_per_window):.6f}(+/-{np.std(inf_time_per_window):.6f})")
            
            for metric in metrics_list:

                print(f'Evaluation with {metric} metrics:\n')
                for metric_name, metrics in performance_metrics[metric].items():
                        print(f"{metric_name} : {np.mean(metrics):.4f}(+/-{np.std(metrics):.4f})")
                print('---' * 20)

            print(f'Experiments finished !')


def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    all_models = ['PCA', 'OC-SVM', 'IForest', 'AE', 'LSTM-VAE', 'DAGMM', 'Deep-SVDD', 'USAD']
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=10, 
                        help='The window size. Default is 10.')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='The threshold of anomalous observations to consider in a window. Default is 0.8.')
    parser.add_argument('--contamination-ratio', type=float, default=0.0, 
                        help='The contamination ratio. Default is 0.')
    parser.add_argument('--n_runs', type=int, default=5, 
                        help='The number of time each experiment is runned. Default is 5.')
    parser.add_argument('--models_list', nargs='*', default=all_models, choices=all_models,
                       help='The list of models to evaluate.')
    parser.add_argument('--save-dir', type=str, default='/home/jupyter/scripts/experiments_outs/metric_exps',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='/home/jupyter/datasets/Datasets/outputs_csv/',
                        help='The folder where the data are stored.')
    parser.add_argument('--is-trained', action='store_true',
                        help='If the models are already trained. Default action is false.')
    return parser.parse_args()

def run_all_experiments(args):
    


    save_dir = args.save_dir
    data_dir = args.data_dir
    n_runs = args.n_runs
    window_size = args.window_size
    threshold = args.threshold
    cont_rate = args.contamination_ratio
    models_list = args.models_list
    is_trained = args.is_trained
    print(is_trained)
    # Check if the threshold is between 0 and 1
    if (threshold<=1 and threshold>=0):
        threshold_int = int(window_size*threshold)
        print(threshold_int)
    else:
        raise ValueError("The threshold must be a float between 0 and 1 !")
    
    # # Check if the models list is in the list of all models
    # if not set(models_list).issubset(all_models):
    #     raise ValueError(f'The list of models must be in {all_models}')
        
    # import sys
    #sys.exit()
    
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    print("Start the experiments...")
    # Run the experiments for all the algorithms
    for model_name in models_list:
        metric_experiment(model_name=model_name, save_dir=save_dir, data_dir=data_dir, n_runs=n_runs, 
                        window_size=window_size, cont_rate=cont_rate, threshold=threshold_int, is_trained=is_trained)

    print("All the experiments are completed !")
    
if __name__ == '__main__':
    args = parse_arguments()
    run_all_experiments(args)




