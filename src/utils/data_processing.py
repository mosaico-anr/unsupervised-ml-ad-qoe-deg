# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import os
from itertools import groupby

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader



network_conditions = ['excellent', 'very_good', 'good', 'average', 'bad', 'highway']
cg_platforms = ['std', 'xc', 'gfn', 'psn']

#col_drop = ['count', 'first_frame_received_to_decoded_ms', 'frames_rendered', 'interframe_delay_max_ms', 'current_delay','target_delay_ms', 'decode_delay',
#           'jb_cuml_delay', 'jb_emit_count', 'sum_squared_frame_durations', 'sync_offset_ms', 'total_bps', 'total_decode_time_ms',  'total_frames_duration_ms',
#           'total_freezes_duration_ms', 'total_inter_frame_delay', 'total_pauses_duration_ms', 'total_squared_inter_frame_delay', 
#           'max_decode_ms', 'render_delay_ms', 'min_playout_delay_ms', 'dec_fps', 'ren_fps', 'cdf', 'packetsReceived', 'packetsLost', 'time_ms']
to_keep = ['time_ms', 'decode_delay', 'jitter','jb_delay', 'packetsReceived_count',
            'net_fps', 'height', 'width', 'frame_drop','frames_decoded', 'rtx_bps', 'rx_bps', 'freeze',
            'throughput','rtts']

class WindowAnomaly():
    def __init__(self, window: list, threshold: int=5, pos_val=1., is_consecutive=False):
        self.values = window
        self.pos_val = pos_val
        self.threshold = threshold
        self.is_consecutive = is_consecutive
        # Features for consecutive anomalies
        self.anomalies_start_end = {}
        self.normal_start_end = {}
        self.consecutive_anomalies = {'wad' : [], 'rpa': [], 'pa': self.values}
        
        # Features for anomalous in term of threshold percentage of the window size
        self.threshold_anomalies = {'wad' : [], 'rpa': [], 'pa': self.values}
        
        if self.is_consecutive:
            self._make_anomalies_delimiters()
            self._make_normal_delimiters()
            self._make_consecutive_anomalies()
        else:
            self._make_threshold_anomalies()
        
        
        
    def get_ground_truths(self, window_type='wad'):
        if self.is_consecutive:
            return self.consecutive_anomalies[window_type]
        else:
            return self.threshold_anomalies[window_type]
#         if window_type == 'wad':
#             return self.anomalies_wad
#         elif window_type == 'pa':
#             return self.values
#         elif window_type == 'rpa':
#             return self.anomalies_rpa
            
#         else:
#             raise ValueError(f'The window type {window_type} is not known !')
            
        
    def _make_threshold_anomalies(self):
        count_anomaly = np.count_nonzero(self.values == self.pos_val)
        # print(count_anomaly)
        if count_anomaly >= self.threshold:
            # anomalous window
            self.threshold_anomalies['wad'] = [1]
            self.threshold_anomalies['rpa'] = [1]
        else:
            # normal window
            self.threshold_anomalies['wad'] = [0]
            self.threshold_anomalies['rpa'] = self.values
        
        
        
        
    def _make_anomalies_delimiters(self):
        index_list = list(range(len(self.values)))
        data = zip(self.values, index_list)
        for key, group in groupby(data, lambda x: x[0]):
            label, index = next(group)
            nb_consec = len(list(group))+1
            if label == self.pos_val and nb_consec >=self.threshold:
                self.anomalies_start_end[index] = index+nb_consec
                #self.anomalies_start_end.append((index, index+nb_consec-1))
                #self.anomalies_start.append(index)
                
    def _make_normal_delimiters(self):
        ind = 0
        anomalies_start_list = list(self.anomalies_start_end.keys()) # List of the start of each anomalous seq.
        
        while ind!= len(self.values):
            if ind in self.anomalies_start_end.keys():
                ind = self.anomalies_start_end[ind]
            else:
                norm_start = ind
                if not anomalies_start_list:
                    norm_end = len(self.values)
                    ind = norm_end
                else:
                    norm_end = anomalies_start_list.pop(0)
                    ind = self.anomalies_start_end[norm_end]
                self.normal_start_end[norm_start] = norm_end               
            
    
    def _make_consecutive_anomalies(self):
        ind = 0
        
        while ind != len(self.values):
            if ind in self.anomalies_start_end.keys():
                self.consecutive_anomalies['wad'].append(1)
                self.consecutive_anomalies['rpa'].append(1)
                # self.anomalies_wad.append(1)
                # self.anomalies_rpa.append(1)
                ind = self.anomalies_start_end[ind]
            elif ind in self.normal_start_end.keys():
                self.consecutive_anomalies['wad'].append(0)
                self.consecutive_anomalies['rpa'].extend(self.values[ind:self.normal_start_end[ind]])
                # self.anomalies_wad.append(0)
                # self.anomalies_rpa.extend(self.values[ind:self.normal_start_end[ind]])
                ind = self.normal_start_end[ind]
                
            # else:
            #     self.anomalies_adjust.append(self.values[ind])
            #     ind+= 1
        #self.anomalies_adjust = np.array(self.anomalies_adjust)
    
    def compute_window_wised_pred(self, pred_vals, window_type='wad'):
        if self.is_consecutive:
            return self._make_consecutive_preds(pred_vals, window_type=window_type)
        else:
            return self._make_threshold_preds(pred_vals, window_type=window_type)
        
    def _make_threshold_preds(self, pred_vals, window_type='wad'):
        preds_adjust = []
        if self.get_ground_truths('wad') == [0]:
            # If we have a normmal window
            if window_type in ['rpa', 'pa']:
                preds_adjust = pred_vals
            elif window_type == 'wad':
                # Check if the number of detected anomalous >= threshold
                count_an = np.count_nonzero(np.array(pred_vals) == self.pos_val)
                if count_an >= self.threshold:
                    preds_adjust = [1]
                else:
                    preds_adjust = [0]
            else:
                raise ValueError(f'The window type {window_type} is not known !')
                
        else:
            # It is an anomalous window
            anomal_ind = [i for i, an in enumerate(self.values) if an==self.pos_val]
            preds_anom = np.array(pred_vals)[anomal_ind] if window_type in ['rpa', 'pa'] else pred_vals
            if self.pos_val in preds_anom: #  One anomaly were correctly predicted
                if window_type == 'rpa':
                    preds_adjust = [1]
                elif window_type == 'pa':
                    pred_arr = np.array(pred_vals)
                    pred_arr[anomal_ind] = 1
                    preds_adjust = pred_arr.tolist()
                elif window_type == 'wad':
                    # Check if the number of detected anomalous >= threshold
                    count_an = np.count_nonzero(preds_anom == self.pos_val)
                    if count_an >= self.threshold:
                        preds_adjust = [1]
                    else:
                        preds_adjust = [0]
                else:
                    raise ValueError(f'The window type {window_type} is not known !')
            else:
                # No anomaly correctly predicted
                if window_type == 'rpa':
                    preds_adjust = [0]
                elif window_type == 'pa':
                    pred_arr = np.array(pred_vals)
                    pred_arr[anomal_ind] = 0
                    preds_adjust = pred_arr.tolist()
                elif window_type == 'wad':
                    preds_adjust = [0]
                else:
                    raise ValueError(f'The window type {window_type} is not known !')
        #assert(len(preds_adjust)) == len(self.get_ground_truths(window_type=window_type))
        return preds_adjust
    
                
    def _make_consecutive_preds(self, pred_vals, window_type='wad'):
        preds_adjust = []
        # if not self.anomalies_start_end:
        #     # No anomalous seq in the true values => compare as usual
        #     # TODO
        #     # preds_adjust = pred_vals #np.array(pred_vals)
        #     preds_adjust = [0]
        # else:
        #     # Loop in the list of anomalous seq (start, end)
        ind= 0
        while (ind != len(pred_vals)):

            # if there is consecutive anomalies
            if ind in self.anomalies_start_end.keys():
                start, end = ind, self.anomalies_start_end[ind]
                pred_win = pred_vals[start:end]
                # Test if an anomalous seq at least equal to the threshold is detected
                count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in groupby(pred_win) if val==self.pos_val]
                count_val = [tup_count[1] for tup_count in count_consec_anomaly]
                if not count_val:
                    max_consec_anomal = 0
                else:
                    max_consec_anomal = max(count_val)

                if window_type == 'wad' :
                    if max_consec_anomal >= self.threshold:
                        # The anomalous seq is correctly detected
                        preds_adjust.append(1)

                    else:
                        # The anomalous seq is not correctly detected
                        preds_adjust.append(0)
                elif window_type == 'pa':
                    # Applied the point-adjust approach
                    if max_consec_anomal >= 1:
                        # The anomalous seq is correctly detected if one anomalous obs. is correctly detected
                        preds_adjust.extend([1]*len(pred_win))

                    else:
                        # The anomalous seq is not correctly detected
                        preds_adjust.extend([0]*len(pred_win))
                elif window_type == 'rpa':
                    # Apply the revised point-adjust
                    if max_consec_anomal >= 1:
                        # The anomalous seq is correctly detected if one anomalous obs. is correctly detected
                        preds_adjust.append(1)

                    else:
                        # The anomalous seq is not correctly detected
                        preds_adjust.append(0)
                else:
                    raise ValueError(f'The window type {window_type} is not known !')

                ind = end
            # single anomaly => compare as usual
            # else:
            #     preds_adjust.append(pred_vals[ind])
            #     ind += 1
            elif ind in self.normal_start_end.keys():
                start, end = ind, self.normal_start_end[ind]
                pred_win = pred_vals[start:end]
                # print(start, end)


                # Test if an anomalous seq at least equal to the threshold is detected
                count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in groupby(pred_win) if val==self.pos_val]
                count_val = [tup_count[1] for tup_count in count_consec_anomaly]
                if not count_val:
                    max_consec_anomal = 0
                else:
                    max_consec_anomal = max(count_val)

                if window_type == 'wad' :
                    if max_consec_anomal == len(pred_win) :
                        # The normal seq is not correctly detected
                        preds_adjust.append(1)

                    else:
                        # The norm seq is correctly detected
                        preds_adjust.append(0)
                elif window_type in ['pa', 'rpa']:
                    # Applied the point-adjust approach
                    # single anomaly => compare as usual
                    # else:
                    preds_adjust.extend(pred_win)

                else:
                    raise ValueError(f'The window type {window_type} is not known !')
                ind = end
                    
                    
        assert(len(preds_adjust)) == len(self.get_ground_truths(window_type=window_type))          
        # if window_type in ['rpa', 'wad']:
        #     assert(len(preds_adjust) == len(self.anomalies_adjust))
        # elif window_type == 'pa':
        #     assert(len(preds_adjust) == len(self.values))
        return preds_adjust

def read_csv_files(path):
    """Read and load the csv files.

    Args:
        path (str): The path of a csv file data.

    Returns:
            The pandas Dataframe.
    """
    time_step = 5
    df = pd.read_csv(path)
    df['packetsReceived_count'] = [0.0] + [curr - previous for previous,curr in zip(df['packetsReceived'].values, df['packetsReceived'].iloc[1:].values)]
    df['time_ms'] = pd.to_timedelta(df['time_ms'], unit='ms')
    df = df[to_keep]
    df = df.dropna(axis=0)
    df = df.set_index('time_ms').resample(f"{time_step}ms").last()
    df = df.reset_index()
    return df.dropna(axis=0)


def get_data_dict(data_path):
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
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df_dicts[ntw_cnd] = df
    return df_dicts

def label_on_conditions(df):
    """The labels creation of the datasets according to CG platforms recommendations.

    Args:
        df (pd.Dataframe): The raw dataframe.
    """
    conditions = [
        (df['height'] == 720.0) | (df['freeze'] == 1.0) | (df['net_fps'] < 60.0),
        (df['height'] == 1080.0) & (df['freeze'] == 0.0) & (df['net_fps'] >= 60.0)
        ]
    values = [1.0, 0.0]
    df['anomaly'] = np.select(conditions, values)
    
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
    [excluded_starts.extend(range((start - window_size + 1), start)) for start in start_discont if start > window_size]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - window_size + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + window_size] for i in seq_starts])
    return x_seqs

def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits,
                          seed: int, shuffle: bool = False, usetorch = True):
    """[summary]

    Args:
        x_seqs (np.ndarray): [description]
        batch_size (int): [description]
        splits ([type]): [description]
        seed (int): [description]
        shuffle (bool, optional): [description]. Defaults to False.
        usetorch (bool, optional): [description]. Defaults to True.

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
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]], batch_size=batch_size,
            drop_last=False, pin_memory=True, shuffle=False) for i in range(len(splits))])
        return loaders
    else:
        # datasets = tuple([x_seqs[split_points[i]: 
        #     (split_points[i] + (split_points[i+1]-split_points[i])//batch_size*batch_size)] 
        #     for i in range(len(splits))])
        datasets = tuple([x_seqs[split_points[i]:split_points[i+1]]
            for i in range(len(splits))])
        return datasets
    

def identify_window_anomaly(window, window_size, threshold, pos_val=1.0, is_consecutive=False):
    if is_consecutive:
        count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in groupby(window) if val==pos_val]
        count_val = [tup_count[1] for tup_count in count_consec_anomaly]
    
    #print(count_consec_anomaly)
        if not count_val:
            total_anomal_obs, max_consec_anomal = 0, 0
        else:
            total_anomal_obs, max_consec_anomal = sum(count_val), max(count_val)
    else:
        total_anomal_obs = np.count_nonzero(window == pos_val)
    #return total_anomal_obs, max_consec_anomal
    #print(total_anomal_obs, window_size//2, max_consec_anomal, threshold)
    # an = []
    #for v in count_val:
    if (total_anomal_obs >= threshold):  #(v >= threshold): # or
            #an.append(1)
        return 1
    else:
        return 0
#     if an:
#         return an
#     else:
#         return [0]
        
        
def window_processing(cg_df: pd.DataFrame, window_size=10, threshold=10, is_consecutive=False):
    # train_cg_1.columns.get_loc('anomaly')
    cg_win = get_sub_seqs(cg_df.values, window_size)
    x = []
    y = []
    for i in range(cg_win.shape[0]):
        x.append(cg_win[i,:,0:14])
        y.append(cg_win[i,:,-1])
        
    cg_x = np.stack(x, axis=0)
    cg_y = np.stack(y, axis=0)
    
    window_labels = []
    #thresh = 1
    for i in range(cg_y.shape[0]):
        win = cg_y[i]
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size, threshold=threshold, is_consecutive=is_consecutive)))
    values, counts = np.unique(window_labels, return_counts=True)
    print(values, counts)
    window_labels = np.array(window_labels)
    return {'X_array': cg_x, 'y_labels': cg_y, 'window_labels': window_labels}

def get_all_data(df_dicts, window_size=10, threshold=10, is_consecutive=False):
    df_list = []
    for ntw_cnd in network_conditions:
        df = df_dicts[ntw_cnd].drop('time_ms', axis=1)
        #print(df.isna().sum().sum())
        df_list.append(df)
        
    categorical_col = ['height', 'width', 'freeze']
    categorical_index = []
    for col_name in categorical_col:
        categorical_index.append(df_list[0].columns.get_loc(col_name))

    df_dicts_list = []
    for i in range(len(df_list)):
        df_dicts_list.append(window_processing(df_list[i], window_size=window_size, threshold=threshold, is_consecutive=is_consecutive))
    X = [v['X_array'] for v in df_dicts_list]
    X = np.concatenate(X, axis=0)
    y = [v['y_labels'] for v in df_dicts_list]
    y = np.concatenate(y, axis=0)
    y_window = [v['window_labels'] for v in df_dicts_list]
    y_window = np.concatenate(y_window, axis=0)


    # Concatenate all X et y data
    y = y[:,:, np.newaxis]
    all_data = np.concatenate([X, y], axis=-1)
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
    contamination_df, test_anomaly = anomaly_arr[contamination_set_indices,:,:], anomaly_arr[test_anomaly_indices,:,:]
    
    
    # train = pd.concat([train_normal, contamination_df], axis=0, ignore_index=True)
    # train = train.sample(frac=1, ignore_index=True, random_state=seed)
    X_train = np.concatenate([train_normal, contamination_df], axis=0)
    train = np.take(X_train,np.random.rand(X_train.shape[0]).argsort(),axis=0,out=X_train)
    
    # test = pd.concat([test_normal, test_anomaly], axis=0, ignore_index=True)
    # test = test.sample(frac=1, ignore_index=True, random_state=seed)
    
    X_test = np.concatenate([test_normal, test_anomaly], axis=0)
    test = np.take(X_test,np.random.rand(X_test.shape[0]).argsort(),axis=0,out=X_test)
    
    return train, test

def split_contamination_data(train_arr, contamination_ratio, window_size=10, seed=42, threshold=10, is_consecutive=False):
    """The creation of the training data with contamination_ratio % of anomalous samples.

    Args:
        train_arr (pd.Dataframe)    : The training dataframe. 
        contamination_ratio (float) : The contamination ratio, between 0 and 1.
        seed (int, optional)        : The random generator seed. Defaults to 42.

    Raises:
        ValueError                  : If the contamination ratio is not between 0 and 1.

    Returns:
            The training data with contaminated data.
    """
    
    # Get the window labels
    window_labels = []
    #thresh = window_size//2
    y = train_arr[:,:,-1]
    for i in range(y.shape[0]):
        win = y[i]
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size, threshold=threshold, is_consecutive=is_consecutive)))
    values, counts = np.unique(window_labels, return_counts=True)
    print(f"Number of normal/anomalous windows: {counts[0]}/{counts[1]}")
    window_labels = np.array(window_labels)
        
    anomaly_index_list = np.where(window_labels==1.0)[0]
    #print(len(anomaly_index_list))
    
    if contamination_ratio==0:
        train_X = np.delete(train_arr, anomaly_index_list, axis=0)
        train_win_y = np.delete(window_labels, anomaly_index_list, axis=0)
        train_y = np.delete(y, anomaly_index_list, axis=0)
        print(f"Kept only the normal windows in the training set.")
        return {'data' : train_X[:,:, :14], 'window_labels': train_win_y, 'labels': train_y}
    elif contamination_ratio==1:
        print(f"Kept all the anomalous windows in the training set for contamination.")
        return {'data' : train_arr[:,:, :14], 'window_labels': window_labels, 'labels': y}
    
    elif contamination_ratio < 1:
        
        shuffled_indices = np.random.RandomState(seed=seed).permutation(len(anomaly_index_list))
        contamination_set_size = int(len(anomaly_index_list) * contamination_ratio)
        print(f"Kept {contamination_set_size} over {counts[1]} in the training set for contamination.")
        


        # Get the indices of the anomaly to keep
        #anomaly_indices_to_keep = anomaly_index_list[shuffled_indices[:contamination_set_size]]
        anomaly_indices_to_drop = anomaly_index_list[shuffled_indices[contamination_set_size:]]
        
        train_X = np.delete(train_arr, anomaly_indices_to_drop, axis=0)
        train_win_y = np.delete(window_labels, anomaly_indices_to_drop, axis=0)
        train_y = np.delete(y, anomaly_indices_to_drop, axis=0)
        
        return {'data' : train_X[:,:, :14], 'window_labels': train_win_y, 'labels': train_y}
    else:
        raise ValueError('Contamination ratio must be between 0 and 1 !')

def get_X_y_data(all_data, window_size=10, threshold=10, is_consecutive=False):
    
    # Get the window labels
    window_labels = []
    window_adjust = []
    #thresh = window_size//2
    y = all_data[:,:,-1]
    for i in range(y.shape[0]):
        win = y[i]
        win_obj = WindowAnomaly(window=win, threshold=threshold, is_consecutive=is_consecutive)
        window_adjust.append(win_obj)
        window_labels.append(int(identify_window_anomaly(win, window_size=window_size, threshold=threshold, is_consecutive=is_consecutive)))
    values, counts = np.unique(window_labels, return_counts=True)
    #print(values, counts)
    print(f"Number of normal/anomalous windows: {counts[0]}/{counts[1]}")
    window_labels = np.array(window_labels)
    #window_adjust = np.array(window_adjust)
    
    return {'data' : all_data[:,:, :14], 'window_labels': window_labels, 'labels': y, 'window_adjust': window_adjust} 

def data_processing_random(normal_arr, anomaly_arr, contamination_ratio, window_size=10, seed=42, threshold=10, is_consecutive=False):
    """The "mixed-datasets" splitting strategy.

    Args:
        normal_df (pd.Dataframe)        : The dataframe of normal observationsconcatenated.
        anomaly_df (pd.Dataframe)       : The dataframe of anomalous observations concatenated.
        contamination_ratio (float)     : The contamination ratio. Must be between 0 and 1. 
        window_size (int, optional)     : The window size. Defaults to 10.
        seed (int, optional)            : The random generator seed. Defaults to 42.

    Returns:
            The data dict with the train/test and the labels.
    """
    train, test = train_test_split(normal_arr, anomaly_arr, seed=seed)
    
    train_d = split_contamination_data(train, contamination_ratio=contamination_ratio, seed=seed, threshold=threshold, is_consecutive=is_consecutive)
    
    test_d = get_X_y_data(test, window_size=window_size, threshold=threshold, is_consecutive=is_consecutive)
    
    d = {}
    d['train'] = train_d
    d['test'] = test_d
    return d

def data_processing(data_path, window_size=10, contamination_ratio=0.5, seed=42, threshold=10, is_consecutive=False):
    """[summary]

    Args:
        data_path (str): The data csv folder path.

    Returns:
    """
    df_dicts = get_data_dict(data_path)
    for _, df in df_dicts.items():
        label_on_conditions(df)
    
#     # Concat the dataframes
#     train_cg_1 = df_dicts['excellent']
#     # train_cg_2 = df_dicts['very_good']
#     # train_cg_3 = df_dicts['good']

#     categorical_col = ['height', 'width', 'freeze']
#     categorical_index = []
#     for col_name in categorical_col:
#         categorical_index.append(train_cg_1.columns.get_loc(col_name))
    
    
#     train_cg_1 = train_cg_1.drop('time_ms', axis=1)
#     train_cg_2 = train_cg_2.drop('time_ms', axis=1)
#     train_cg_3 = train_cg_3.drop('time_ms', axis=1)
    
#     test_cg_1 = df_dicts['average']
#     test_cg_2 = df_dicts['bad']
#     test_cg_3 = df_dicts['highway']
    
#     test_cg_1 = test_cg_1.drop('time_ms', axis=1)
#     test_cg_2 = test_cg_2.drop('time_ms', axis=1)
#     test_cg_3 = test_cg_3.drop('time_ms', axis=1)
    
    normal_arr, anomaly_arr, categorical_index = get_all_data(df_dicts, window_size=window_size, threshold=threshold,
                                                             is_consecutive=is_consecutive)

    train_test_data_dict = data_processing_random(normal_arr, anomaly_arr, contamination_ratio, threshold=threshold,
                                                  window_size=window_size, seed=seed, is_consecutive=is_consecutive)
    
    return train_test_data_dict, categorical_index