"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import os
from itertools import groupby

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.window import WindowAnomaly



network_conditions = ['excellent', 'very_good', 'good', 'average', 'bad', 'highway']
cg_platforms = ['std', 'xc', 'gfn', 'psn']

#col_drop = ['count', 'first_frame_received_to_decoded_ms', 'frames_rendered',
#           interframe_delay_max_ms', 'current_delay','target_delay_ms', 'decode_delay',
#           'jb_cuml_delay', 'jb_emit_count', 'sum_squared_frame_durations', 'sync_offset_ms',
#           'total_bps', 'total_decode_time_ms', 'total_frames_duration_ms',
#           'total_freezes_duration_ms', 'total_inter_frame_delay', 'total_pauses_duration_ms',
#           'total_squared_inter_frame_delay', 'max_decode_ms', 'render_delay_ms',
#           'min_playout_delay_ms', 'dec_fps', 'ren_fps', 'cdf', 'packetsReceived',
#           'packetsLost', 'time_ms']
to_keep = ['time_ms', 'decode_delay', 'jitter','jb_delay', 'packetsReceived_count',
            'net_fps', 'height', 'width', 'frame_drop','frames_decoded', 'rtx_bps', 'rx_bps',
            'freeze','throughput','rtts']


def read_csv_files(path):
    """Read and load the csv files.

    Args:
        path (str): The path of a csv file data.

    Returns:
            The pandas Dataframe.
    """
    time_step = 5
    file_df = pd.read_csv(path)
    file_df['packetsReceived_count'] = [0.0] + [curr - previous for previous,curr in
                zip(file_df['packetsReceived'].values, file_df['packetsReceived'].iloc[1:].values)]
    file_df['time_ms'] = pd.to_timedelta(file_df['time_ms'], unit='ms')
    file_df = file_df[to_keep]
    file_df = file_df.dropna(axis=0)
    file_df = file_df.set_index('time_ms').resample(f"{time_step}ms").last()
    file_df = file_df.reset_index()
    return file_df.dropna(axis=0)


def get_std_data_dict(data_path):
    """Load and merge the data from CG outputs.

    Args:
        data_path (str): The data folder path.

    Returns:
            The dict with all the data.
    """
    csv_path_1 = os.path.join(data_path,'racing_std_1/')
    csv_path_2 = os.path.join(data_path, 'racing_std_2/')
    df_dicts = {}

    for ntw_cnd in network_conditions:
        path_1 = f"{csv_path_1+ntw_cnd}.csv"
        path_2 = f"{csv_path_2+ntw_cnd}.csv"

        df1 = read_csv_files(path_1)
        df2 = read_csv_files(path_2)
        file_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df_dicts[ntw_cnd] = file_df
    return df_dicts

def get_data_dict(data_path, platform='gfn'):
    """Get the CG data in a dict.

    Args:
        data_path (str): The path to the folder of csv files.
        platform (str, optional): CG platform to load. Defaults to 'gfn'.

    Returns:
        dict: Data dict.
    """

    csv_path = os.path.join(data_path, f'racing_{platform}_1/')
    df_dicts = {}

    for ntw_cnd in network_conditions:
        path = f"{csv_path+ntw_cnd}.csv"
        cg_df = read_csv_files(path)
        df_dicts[ntw_cnd] = cg_df

    return df_dicts

def label_on_conditions(cg_df, platform='std'):
    """The labels creation of the datasets according to CG platforms recommendations.

    Args:
        cg_df (pd.Dataframe): The raw dataframe.
        platform (str): The CG platform.
    """
    if platform in ['std', 'xc']:
        good_resolution, bad_resolution = 1080.0, 720.0
    elif platform == 'gfn':
        good_resolution, bad_resolution = 768.0, 576.0
    else:
        raise ValueError(f'The platform {platform} is unknown !')

    conditions = [
    (cg_df['height'] == bad_resolution) | (cg_df['freeze'] == 1.0) | (cg_df['net_fps'] < 60.0),
    (cg_df['height'] == good_resolution) & (cg_df['freeze'] == 0.0) & (cg_df['net_fps'] >= 60.0)
        ]
    values = [1.0, 0.0]
    cg_df['anomaly'] = np.select(conditions, values)

def get_sub_seqs(x_arr, window_size, stride=1, start_discont=np.array([])):
    """Process the data into moving window.

    Args:
        x_arr (np.array)                    : The data arrays. 
        window_size (int)                   : The window size.
        stride (int, optional)              : The stride value. Defaults to 1.
        start_discont (np.array, optional)  : Defaults to np.array([]).

    Returns:
    """
    excluded_starts = []
    for start in start_discont:
        if start > window_size:
            excluded_starts.extend(range((start - window_size + 1), start))
    # [excluded_starts.extend(range((start - window_size + 1), start))
    # for start in start_discont if start > window_size]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - window_size + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + window_size] for i in seq_starts])
    return x_seqs

def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits,
                        seed: int, shuffle: bool = False, usetorch = True):
    """ Create data loaders.

    Args:
        x_seqs (np.ndarray): Window arrays.
        batch_size (int): Data loaders batch size.
        splits ([type]): [description]
        seed (int): Random seed value.
        shuffle (bool, optional): If the data should be shuffle. Defaults to False.
        usetorch (bool, optional): If should create PyTorch dataloaders. Defaults to True.

    Returns:
        [type]: [description]
    """
    if np.sum(splits) != 1:
        scale_factor = np.sum(splits)
        splits = [fraction/scale_factor for fraction in splits]
    if shuffle:
        np.random.seed(seed)
        x_seqs = x_seqs[np.random.permutation(len(x_seqs))]
        np.random.seed()
    split_points = [0]
    for i in range(len(splits)-1):
        split_points.append(split_points[-1] + int(splits[i]*len(x_seqs)))
    split_points.append(len(x_seqs))
    if usetorch:
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]],
                                    batch_size=batch_size, drop_last=False, pin_memory=True,
                                    shuffle=False) for i in range(len(splits))])
        return loaders

    # datasets = tuple([x_seqs[split_points[i]:
    #     (split_points[i] + (split_points[i+1]-split_points[i])//batch_size*batch_size)]
    #     for i in range(len(splits))])
    datasets = tuple([x_seqs[split_points[i]:split_points[i+1]]
        for i in range(len(splits))])
    return datasets


def identify_window_anomaly(window, window_size, threshold, pos_val=1.0, is_consecutive=False):
    """_summary_

    Args:
        window (_type_): _description_
        window_size (_type_): _description_
        threshold (_type_): _description_
        pos_val (float, optional): _description_. Defaults to 1.0.
        is_consecutive (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # print(threshold, window_size)
    if threshold > window_size:
        raise ValueError("The threshold value is above the window size")
    if is_consecutive:
        count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in
                                groupby(window) if val==pos_val]
        count_val = [tup_count[1] for tup_count in count_consec_anomaly]

    #print(count_consec_anomaly)
        if not count_val:
            total_anomal_obs, _ = 0, 0
        else:
            total_anomal_obs, _ = sum(count_val), max(count_val)
    else:
        total_anomal_obs = np.count_nonzero(window == pos_val)
    #return total_anomal_obs, max_consec_anomal
    #print(total_anomal_obs, window_size//2, max_consec_anomal, threshold)
    # an = []
    #for v in count_val:
    if total_anomal_obs >= threshold:  #(v >= threshold): # or
            #an.append(1)
        return 1
    return 0


def window_processing(cg_df: pd.DataFrame, window_size=10, threshold=10, is_consecutive=False):
    """Process and create window arrays.

    Args:
        cg_df (pd.DataFrame): CG dataframe.
        window_size (int, optional): Size of the window. Defaults to 10.
        threshold (int, optional): Threshold for anomalies. Defaults to 10.
        is_consecutive (bool, optional): If consider only consecutive anomalies.
                                        Defaults to False.

    Returns:
        dict: Data dicts.
    """
    cg_win = get_sub_seqs(cg_df.values, window_size)
    x_list = []
    y_list = []
    for i in range(cg_win.shape[0]):
        x_list.append(cg_win[i,:,0:14])
        y_list.append(cg_win[i,:,-1])

    cg_x = np.stack(x_list, axis=0)
    cg_y = np.stack(y_list, axis=0)

    window_labels = []
    #thresh = 1
    for i in range(cg_y.shape[0]):
        win = cg_y[i]
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size,
                                                        threshold=threshold,
                                                        is_consecutive=is_consecutive)))
    values, counts = np.unique(window_labels, return_counts=True)
    print(values, counts)
    window_labels = np.array(window_labels)
    return {'X_array': cg_x, 'y_labels': cg_y, 'window_labels': window_labels}

def get_all_data(df_dicts, window_size=10, threshold=10, is_consecutive=False):
    """Get normal and anomalous arrays.

    Args:
        df_dicts (dict): Dict of CG dataframes.
        window_size (int, optional): Size of the window. Defaults to 10.
        threshold (int, optional): Threshold for anomalies. Defaults to 10.
        is_consecutive (bool, optional): If consider only consecutive anomalies.
                                        Defaults to False.

    Returns:
        _type_: _description_
    """
    df_list = []
    for ntw_cnd in network_conditions:
        df_cg = df_dicts[ntw_cnd].drop('time_ms', axis=1)
        #print(df.isna().sum().sum())
        df_list.append(df_cg)

    categorical_col = ['height', 'width', 'freeze']
    categorical_index = []
    for col_name in categorical_col:
        categorical_index.append(df_list[0].columns.get_loc(col_name))

    df_dicts_list = []
    for cg_df in df_list:
        df_dicts_list.append(window_processing(cg_df, window_size=window_size,
                                            threshold=threshold,
                                            is_consecutive=is_consecutive))
    x_arr = [v['X_array'] for v in df_dicts_list]
    x_arr = np.concatenate(x_arr, axis=0)
    y_arr = [v['y_labels'] for v in df_dicts_list]
    y_arr = np.concatenate(y_arr, axis=0)
    y_window = [v['window_labels'] for v in df_dicts_list]
    y_window = np.concatenate(y_window, axis=0)


    # Concatenate all X et y data
    y_arr = y_arr[:,:, np.newaxis]
    all_data = np.concatenate([x_arr, y_arr], axis=-1)
    normal_arr = all_data[y_window==0.0]
    anomaly_arr = all_data[y_window==1.0]
    return normal_arr, anomaly_arr, categorical_index


def train_test_split(normal_arr, anomaly_arr, seed=42):
    """Split the data into train-test set.

    Args:
        normal_arr (pd.Dataframe)        : The dataframe of normal observations.
        anomaly_arr (pd.Dataframe)       : The dataframe of anomalous observations.
        seed (int, optional)            : The random seed generator. Defaults to 42.

    Returns:
    """
    # Split normal df in 50 - 50 randomly
    shuffled_indices = np.random.RandomState(seed=seed).permutation(len(normal_arr))
    test_ratio = 0.5
    test_set_size = int(len(normal_arr) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_normal, test_normal = normal_arr[train_indices,:,:], normal_arr[test_indices,:,:]

    # Split anomaly test df and contamination set
    shuffled_indices = np.random.RandomState(seed=seed).permutation(len(anomaly_arr))
    contamination_ratio = 0.4
    contamination_set_size = int(len(anomaly_arr) * contamination_ratio)
    contamination_set_indices = shuffled_indices[:contamination_set_size]
    test_anomaly_indices = shuffled_indices[contamination_set_size:]
    contamination_df, test_anomaly = anomaly_arr[contamination_set_indices,:,:], \
                                        anomaly_arr[test_anomaly_indices,:,:]


    # train = pd.concat([train_normal, contamination_df], axis=0, ignore_index=True)
    # train = train.sample(frac=1, ignore_index=True, random_state=seed)
    x_train = np.concatenate([train_normal, contamination_df], axis=0)
    train = np.take(x_train,np.random.rand(x_train.shape[0]).argsort(),axis=0,out=x_train)

    # test = pd.concat([test_normal, test_anomaly], axis=0, ignore_index=True)
    # test = test.sample(frac=1, ignore_index=True, random_state=seed)

    x_test = np.concatenate([test_normal, test_anomaly], axis=0)
    test = np.take(x_test,np.random.rand(x_test.shape[0]).argsort(),axis=0,out=x_test)

    return train, test

def split_contamination_data(train_arr, contamination_ratio, window_size=10, seed=42,
                            threshold=10, is_consecutive=False):
    """The creation of the training data with contamination_ratio % of anomalous samples.

    Args:
        train_arr (pd.Dataframe)    : The training dataframe. 
        contamination_ratio (float) : The contamination ratio, between 0 and 1.
        seed (int, optional)        : The random generator seed. Defaults to 42.
        window_size (int, optional) : The window size. Defaults to 10.
        threshold (int, optional)   : The threshold to consider a window, anomalous.
                                        Defaults to 10.
        is_consecutive (bool, optional) : If considering only consecutive anomalies.
                                        Defaults to False.

    Raises:
        ValueError                  : If the contamination ratio is not between 0 and 1.

    Returns:
            The training data with contaminated data.
    """
    # Get the window labels
    window_labels = []
    #thresh = window_size//2
    y_arr = train_arr[:,:,-1]
    for i in range(y_arr.shape[0]):
        win = y_arr[i]
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size,
                            threshold=threshold, is_consecutive=is_consecutive)))
    _, counts = np.unique(window_labels, return_counts=True)
    print(f"Number of normal/anomalous windows: {counts[0]}/{counts[1]}")
    window_labels = np.array(window_labels)

    anomaly_index_list = np.where(window_labels==1.0)[0]
    #print(len(anomaly_index_list))

    if contamination_ratio==0:
        train_x = np.delete(train_arr, anomaly_index_list, axis=0)
        train_win_y = np.delete(window_labels, anomaly_index_list, axis=0)
        train_y = np.delete(y_arr, anomaly_index_list, axis=0)
        print("Kept only the normal windows in the training set.")
        return {'data' : train_x[:,:, :14], 'window_labels': train_win_y, 'labels': train_y}
    elif contamination_ratio==1:
        print("Kept all the anomalous windows in the training set for contamination.")
        return {'data' : train_arr[:,:, :14], 'window_labels': window_labels, 'labels': y_arr}

    elif contamination_ratio < 1:

        shuffled_indices = np.random.RandomState(seed=seed).permutation(len(anomaly_index_list))
        contamination_set_size = int(len(anomaly_index_list) * contamination_ratio)
        print(f"Kept {contamination_set_size} over {counts[1]} \
            in the training set for contamination.")



        # Get the indices of the anomaly to keep
        #anomaly_indices_to_keep = anomaly_index_list[shuffled_indices[:contamination_set_size]]
        anomaly_indices_to_drop = anomaly_index_list[shuffled_indices[contamination_set_size:]]

        train_x = np.delete(train_arr, anomaly_indices_to_drop, axis=0)
        train_win_y = np.delete(window_labels, anomaly_indices_to_drop, axis=0)
        train_y = np.delete(y_arr, anomaly_indices_to_drop, axis=0)

        return {'data' : train_x[:,:, :14], 'window_labels': train_win_y, 'labels': train_y}
    else:
        raise ValueError('Contamination ratio must be between 0 and 1 !')

def get_x_y_data(all_data, window_size=10, threshold=10, wad_threshold=10, is_consecutive=False):
    """Get features, labels, window labels data.

    Args:
        all_data (np.ndarray): Data array
        window_size (int, optional): Window size. Defaults to 10.
        threshold (int, optional): Threshold for anomalies. Defaults to 10.
        wad_threshold (int, optional): Threshold for WAD approach. Defaults to 10.
        is_consecutive (bool, optional): If considering only consecutive anomalies.
                                        Defaults to False.

    Returns:
        dict: Data dicts.
    """

    # Get the window labels
    window_labels = []
    window_adjust = []
    #thresh = window_size//2
    y_arr = all_data[:,:,-1]
    for i in range(y_arr.shape[0]):
        win = y_arr[i]
        win_obj = WindowAnomaly(window=win, threshold=wad_threshold,
                                is_consecutive=is_consecutive)
        window_adjust.append(win_obj)
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size,
                                                        threshold=threshold,
                                                        is_consecutive=is_consecutive)))
    _, counts = np.unique(window_labels, return_counts=True)
    #print(values, counts)
    print(f"Number of normal/anomalous windows: {counts[0]}/{counts[1]}")
    window_labels = np.array(window_labels)
    #window_adjust = np.array(window_adjust)

    return {'data' : all_data[:,:, :14], 'window_labels': window_labels, 'labels': y_arr,
            'window_adjust': window_adjust} 

def data_processing_random(normal_arr, anomaly_arr, contamination_ratio, window_size=10,
                        seed=42, threshold=10, wad_threshold=10, is_consecutive=False):
    """Data plitting strategy.

    Args:
        normal_df (pd.Dataframe)        : The dataframe of normal observationsconcatenated.
        anomaly_df (pd.Dataframe)       : The dataframe of anomalous observations concatenated.
        contamination_ratio (float)     : The contamination ratio. Must be between 0 and 1. 
        window_size (int, optional)     : The window size. Defaults to 10.
        seed (int, optional)            : The random generator seed. Defaults to 42.
        threshold (int, optional)       : Threshold to consider anomalies. Defaults to 10.
        wad_threshold (int, optional)   : WAD approach threshold. Defaults to 10.
        is_consecutive (bool, optional) : If considering only consecutive anomalies.
                                            Defaults to False.

    Returns:
            The data dict with the train/test and the labels.
    """
    train, test = train_test_split(normal_arr, anomaly_arr, seed=seed)

    train_d = split_contamination_data(train, window_size=window_size,
                                    contamination_ratio=contamination_ratio,
                                    seed=seed, threshold=threshold,
                                    is_consecutive=is_consecutive)

    test_d = get_x_y_data(test, window_size=window_size, threshold=threshold,
                        wad_threshold=wad_threshold,
                        is_consecutive=is_consecutive)

    data_dict = {}
    data_dict['train'] = train_d
    data_dict['test'] = test_d
    return data_dict

def data_processing(data_path, window_size=10, contamination_ratio=0.5, seed=42,
                    wad_threshold=10,
                    threshold=10,
                    is_consecutive=False,
                    platform='std'):
    """Data processing function.

    Args:
        data_path (str): Data folder path.
        window_size (int, optional): Window size. Defaults to 10.
        contamination_ratio (float, optional): Contamination ratio. Defaults to 0.5.
        seed (int, optional): Random seed. Defaults to 42.
        threshold (int, optional): Threshold for anomaly window. Defaults to 10.
        is_consecutive (bool, optional): If considering consecutive anomalies. Defaults to False.
        platform (str, optional): CG platforms. Defaults to 'std'.

    Returns:
        tuple (dict,): (data_dict, categorical_index)
    """
    if platform in ['gfn', 'xc']:
        df_dicts = get_data_dict(data_path, platform=platform)
    elif platform =='std':
        df_dicts = get_std_data_dict(data_path)
    else:
        print("Unknow platform ! Load stadia CG outputs..")
        df_dicts = get_std_data_dict(data_path)
    for _, cg_df in df_dicts.items():
        label_on_conditions(cg_df, platform=platform)

    normal_arr, anomaly_arr, categorical_index = get_all_data(df_dicts, window_size=window_size,
                                                            threshold=threshold,
                                                            is_consecutive=is_consecutive)

    train_test_data_dict = data_processing_random(normal_arr, anomaly_arr, contamination_ratio,
                                                threshold=threshold,
                                                wad_threshold=wad_threshold,
                                                window_size=window_size,
                                                seed=seed,
                                                is_consecutive=is_consecutive)

    return train_test_data_dict, categorical_index
