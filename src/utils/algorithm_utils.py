"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import abc
import os
import logging
import random
import pickle

import numpy as np
import torch
from torch.autograd import Variable
#from tensorflow.python.client import device_lib
#import tensorflow as tf


class Algorithm(metaclass=abc.ABCMeta):
    """Algorithm class.

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    def __init__(self, module_name, name, seed):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.prediction_details = {}

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, train_data):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, test_data):
        """
        :return anomaly score
        """

    def get_val_err(self):
        """
        :return: reconstruction error_tc for validation set,
        dimensions of num_val_time_points x num_channels
        Call after training
        """
        return None

    def get_val_loss(self):
        """
        :return: scalar loss after training
        """
        return None


def save_torch_algo(algo: Algorithm, save_dir, torch_model=True):
    """The save model function.

    Args:
        algo (Algorithm)                : The algorithm to save
        save_dir (str)                  : The save folder path.
        torch_model (bool, optional)    : If the model is a torch model. 
                                        Defaults to True.

    Returns:
            The filename of the saved model, the algo config and the additional_params.
    """
    # Check if the folder exist
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Save the model
    saved_model_filename = os.path.join(save_dir, "model")
    if torch_model:
        torch.save(algo.model, saved_model_filename)
    else:
        with open(saved_model_filename+'', "wb") as file:
            pickle.dump(algo.model, file)

    # Save init parameters
    init_params = algo.init_params
    algo_config_filename = os.path.join(save_dir, "init_params")
    with open(algo_config_filename, "wb") as file:
        pickle.dump(init_params, file)

    # Save additional parameters
    additional_params_filename = os.path.join(save_dir, "additional_params")
    additional_params = algo.additional_params
    with open(additional_params_filename, "wb") as file:
        pickle.dump(additional_params, file)

    return saved_model_filename, algo_config_filename, additional_params_filename




def load_torch_algo(algo_class, algo_config_filename, saved_model_filename,
                    additional_params_filename, evaluation=True, torch_model=True):
    """The function to load the saved model.

    Args:
        algo_class (Algorithm)          : The class of the algorithm to load.
        algo_config_filename (str)      : The filename of the algo config.
        saved_model_filename (str)      : The filename of the saved model.
        additional_params_filename (str): The filename of the algo additional params.
        eval (bool, optional)           : If the model is load for evaluation.
                                        Defaults to True.
        torch_model (bool, optional)    : If the model is a torch model.
                                        Defaults to True.

    Returns:
            The algorithm loaded model. 
    """

    with open(os.path.join(algo_config_filename), "rb") as file:
        init_params = pickle.load(file)

    with open(additional_params_filename, "rb") as file:
        additional_params = pickle.load(file)

    # init params must contain only arguments of algo_class's constructor
    algo = algo_class(**init_params)


    if additional_params is not None:
        setattr(algo, "additional_params", additional_params)

    if torch_model:
        device = algo.device
        algo.model = torch.load(saved_model_filename, map_location=device)
        if evaluation:
            algo.model.eval()
    else:
        with open(saved_model_filename, 'rb') as file:
            algo.model = pickle.load(file)
    return algo

class PyTorchUtils(metaclass=abc.ABCMeta):
    """Utils for PyTorch usage.

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0
        self.torch_save = True

    @property
    def device(self):
        """Return Torch device.
        """
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available()
                            and self.gpu is not None else 'cpu')

    def to_var(self, tensor, **kwargs):
        """Return a torch variable.

        Args:
            variable (torch.Tensor): Torch tensor

        Returns:
            torch.Variable.
        """
        tensor = tensor.to(self.device)
        return Variable(tensor, **kwargs)

    def to_device(self, model):
        """Convert the model to the correct device.

        Args:
            model (torch.nn): Torch model.
        """
        model.to(self.device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset the class attributes.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n_count=1):
        """Update the values.

        Args:
            val (float): Current value.
            n_count (int, optional): Number of current value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n_count
        self.count += n_count
        self.avg = self.sum / self.count
