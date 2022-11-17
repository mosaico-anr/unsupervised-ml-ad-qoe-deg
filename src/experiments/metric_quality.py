# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import argparse, os, time
import contextlib

import numpy as np

from src.utils.data_processing import data_processing
from src.utils.evaluation_utils import get_performance
from src.models.random_guessing import Random_guessing
from src.models.zero_rule import ZeroR
from src.models.beta_detector import BetaDetector


all_models = ['random_guessing', 'zeroR']
all_algo = [Random_guessing, ZeroR]
beta_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
# beta_list = [0.1, 0.2]
all_models +=[f'betaDetector_{beta}' for beta in beta_list]
all_algo +=[BetaDetector for _ in beta_list]
assert(len(all_models) == len(all_algo))
models_algo_name_map = {all_models[i]: all_algo[i] for i in range(len(all_models))}
metrics_list = ['pw', 'window_wad', 'window_pa', 'window_rpa']

def metric_quality_experiment(model_name, save_dir, data_dir, n_runs=5, window_size=10, cont_rate=0, threshold=10,
                                 verbose=False, is_consecutive=False):

    
    # model_save_path = f'{save_dir}/{model_name}/'
    # if not (os.path.isdir(model_save_path)):
    #     os.mkdir(model_save_path)

    result_file = f'{save_dir}/{model_name}_results.txt'

    with open(result_file, 'w') as f:
        with contextlib.redirect_stdout(f):

            performance_metrics = {'train_time' : [], 'test_time': [] }
            for met in metrics_list:
                performance_metrics[met] = { 'f1' : [],'recall' : [], 'precision' : [], 'auc' : [],
                                             'mcc' : [], 'acc_anom' : [], 'acc_norm' : [] }
            
            print(f'Begin experiments on {model_name} algorithm...')

            for i in range(n_runs):

                # Get the data
                seed = 42+i
                # print(threshold)
                data_random, categorical_ind = data_processing(data_dir, window_size=window_size, contamination_ratio=cont_rate,
                                                             seed=seed, threshold=threshold, is_consecutive=is_consecutive)

                # Create the model
                model_alg = models_algo_name_map.get(model_name)
                if model_name.startswith('betaDetector_'):
                    _, beta = model_name.split('_')
                    # print(beta)
                    model = model_alg(seed=seed, beta=float(beta))
                else:
                    model = model_alg(seed=seed)
                performance_metrics['train_time'].append(-1)

                # Evaluate on the test set
                start = time.time()
                if model_name.startswith('betaDetector_'):
                    an_dict = model.predict(data_random['test']['labels'])
                else:
                    an_dict = model.predict(data_random['test']['data'])
                test_time = time.time() - start
                performance_metrics['test_time'].append(test_time)
                model_score = an_dict['anomalies_score']

                for metric in metrics_list:
                    if metric.startswith('window'):
                        metric, window_type = metric.split('_')
                    else:
                        window_type = 'wad'

                    #score_norm = model_score
                    y_pred = an_dict['anomalies']

                    #print(score_norm.shape, y_pred.shape)
                    precision, recall, f1, auc, mcc, acc_anom, acc_norm = get_performance(y_pred, y_true=data_random['test']['labels'],
                                                                test_score=model_score, y_win_adjust=data_random['test']['window_adjust'],
                                                                metric_type=metric, window_type=window_type)
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
    # all_models = ['random_guessing', 'zeroR', 'beta_detector']
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

    return parser.parse_args()

def run_all_experiments(args):
    


    save_dir = args.save_dir
    data_dir = args.data_dir
    n_runs = args.n_runs
    window_size = args.window_size
    cont_rate = args.contamination_ratio
    models_list = args.models_list
    threshold = args.threshold

    # threshold_list = [0.8, 0.9, 1]
    
    # # Check if the models list is in the list of all models
    # if not set(models_list).issubset(all_models):
    #     raise ValueError(f'The list of models must be in {all_models}')
        
    # import sys
    #sys.exit()
    
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    print(f"Start the experiments with threshold = {threshold}...")
    # Run the experiments for all the algorithms
    # for threshold in threshold_list:
    threshold_int = int(window_size*threshold)
    thresh_exp_path = f'{save_dir}/exps_{threshold}/'
    if not (os.path.isdir(thresh_exp_path)):
        os.mkdir(thresh_exp_path)
    for model_name in models_list:
        print(f"Start experiments with {model_name}")
        metric_quality_experiment(model_name=model_name, save_dir=thresh_exp_path, data_dir=data_dir, n_runs=n_runs, 
                        window_size=window_size, cont_rate=cont_rate, threshold=threshold_int)
        print(f"Experiments with {model_name} finished !")

    print("All the experiments are completed !")
    
if __name__ == '__main__':
    args = parse_arguments()
    run_all_experiments(args)




