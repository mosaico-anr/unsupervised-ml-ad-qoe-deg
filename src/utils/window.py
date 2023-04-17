"""
Copyright (c) 2022 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

from itertools import groupby

import numpy as np


class WindowAnomaly():
    """Anomaly window class.
    """
    def __init__(self, window: list, threshold: int=5, pos_val=1., is_consecutive=False):
        """Create a window anomaly class.

        Args:
            window (list): Ground-truths point-wise anomalies for a given window.
            threshold (int, optional): Number of anomalies to consider a window as abnormal.
                                        Defaults to 5.
            pos_val (_type_, optional): Positive value . Defaults to 1..
            is_consecutive (bool, optional): If considering consecutive anomalies or not. 
                                            Defaults to False.
        """
        self.values = window
        self.pos_val = pos_val
        self.threshold = threshold
        self.is_consecutive = is_consecutive
        # Features for consecutive anomalies
        self.anomalies_start_end = {}
        self.normal_start_end = {}
        self.consecutive_anomalies = {'wad' : [], 'rpa': [], 'pa': self.values, 'pak': self.values}

        # Features for anomalous in term of threshold percentage of the window size
        self.threshold_anomalies = {'wad' : [], 'rpa': [], 'pa': self.values, 'pak': self.values}

        if self.is_consecutive:
            self._make_anomalies_delimiters()
            self._make_normal_delimiters()
            self._make_consecutive_anomalies()
        else:
            self._make_threshold_anomalies()

    def get_ground_truths(self, window_type='wad'):
        """Get the window anomaly ground truths based on window type.

        Args:
            window_type (str, optional): Window type. Defaults to 'wad'.

        Returns:
            list: Window ground truths.
        """
        if self.is_consecutive:
            return self.consecutive_anomalies[window_type]
        return self.threshold_anomalies[window_type]

    def _make_threshold_anomalies(self):
        """Make threshold for anomalous segments.
        """
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
        """Make delimiters to identify anomalous segments.
        """
        index_list = list(range(len(self.values)))
        data = zip(self.values, index_list)
        for _, group in groupby(data, lambda x: x[0]):
            label, index = next(group)
            nb_consec = len(list(group))+1
            if label == self.pos_val and nb_consec >=self.threshold:
                self.anomalies_start_end[index] = index+nb_consec
                #self.anomalies_start_end.append((index, index+nb_consec-1))
                #self.anomalies_start.append(index)

    def _make_normal_delimiters(self):
        """Make delimiters for normal windows.
        """
        ind = 0
        # List of the start of each anomalous segment
        anomalies_start_list = list(self.anomalies_start_end.keys())
        while ind!= len(self.values):
            if ind in self.anomalies_start_end:
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
        """Build anomaly segments.
        """
        ind = 0
        while ind != len(self.values):
            if ind in self.anomalies_start_end:
                self.consecutive_anomalies['wad'].append(1)
                self.consecutive_anomalies['rpa'].append(1)
                # self.anomalies_wad.append(1)
                # self.anomalies_rpa.append(1)
                ind = self.anomalies_start_end[ind]
            elif ind in self.normal_start_end:
                self.consecutive_anomalies['wad'].append(0)
                self.consecutive_anomalies['rpa'].extend(
                                self.values[ind:self.normal_start_end[ind]]
                                                        )
                # self.anomalies_wad.append(0)
                # self.anomalies_rpa.extend(self.values[ind:self.normal_start_end[ind]])
                ind = self.normal_start_end[ind]

            # else:
            #     self.anomalies_adjust.append(self.values[ind])
            #     ind+= 1
        #self.anomalies_adjust = np.array(self.anomalies_adjust)

    def compute_window_wised_pred(self, pred_vals, window_type='wad'):
        """Compute windows prediction based on window type.

        Args:
            pred_vals (np.array): Prediction arrays.
            window_type (str, optional): Window type. Defaults to 'wad'.

        Returns:
            list: Window-based prediction.
        """
        if self.is_consecutive:
            return self._make_consecutive_preds(pred_vals, window_type=window_type)
        return self._make_threshold_preds(pred_vals, window_type=window_type)

    def _make_threshold_preds(self, pred_vals, window_type='wad'):
        """Compute the prediction from the threshold.

        Args:
            pred_vals (np.array): Prediction array.
            window_type (str, optional): Window type. Defaults to 'wad'.

        Raises:
            ValueError: Window type is not known.

        Returns:
            _type_: _description_
        """
        preds_adjust = []
        if self.get_ground_truths('wad') == [0]:
            # If we have a normmal window
            if window_type in ['rpa', 'pa', 'pak']:
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
            preds_anom = np.array(pred_vals)[anomal_ind] if window_type in ['rpa', 'pa'] \
                                                        else pred_vals
            if self.pos_val in preds_anom: #  One anomaly were correctly predicted
                if window_type == 'rpa':
                    preds_adjust = [1]
                elif window_type == 'pa':
                    pred_arr = np.array(pred_vals)
                    pred_arr[anomal_ind] = 1
                    preds_adjust = pred_arr.tolist()
                elif window_type == 'pak':
                    count_an = np.count_nonzero(preds_anom == self.pos_val)
                    if count_an >= self.threshold:
                        pred_arr = np.array(pred_vals)
                        pred_arr[anomal_ind] = 1
                        preds_adjust = pred_arr.tolist()
                    else:
                        pred_arr = np.array(pred_vals)
                        pred_arr[anomal_ind] = 0
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
                elif window_type in ['pa', 'pak']:
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
        """Compute the prediction adjusted if consecutive considered.

        Args:
            pred_vals (np.array): Prediction array.
            window_type (str, optional): Window type. Defaults to 'wad'.

        Raises:
            ValueError: Window type is not known.

        Returns:
            np.array: Prediction adjusted.
        """
        preds_adjust = []
        # if not self.anomalies_start_end:
        #     # No anomalous seq in the true values => compare as usual
        #     # preds_adjust = pred_vals #np.array(pred_vals)
        #     preds_adjust = [0]
        # else:
        #     # Loop in the list of anomalous seq (start, end)
        ind= 0
        while ind != len(pred_vals):

            # if there is consecutive anomalies
            if ind in self.anomalies_start_end:
                start, end = ind, self.anomalies_start_end[ind]
                pred_win = pred_vals[start:end]
                # Test if an anomalous seq at least equal to the threshold is detected
                count_consec_anomaly = [(val, sum(1 for _ in group)) for val,
                                        group in groupby(pred_win) if val==self.pos_val]
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
                        # The anomalous seq is correctly detected if one anomalous obs.
                        # is correctly detected
                        preds_adjust.extend([1]*len(pred_win))

                    else:
                        # The anomalous seq is not correctly detected
                        preds_adjust.extend([0]*len(pred_win))
                elif window_type == 'rpa':
                    # Apply the revised point-adjust
                    if max_consec_anomal >= 1:
                        # The anomalous seq is correctly detected if one anomalous obs.
                        # is correctly detected
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
            elif ind in self.normal_start_end:
                start, end = ind, self.normal_start_end[ind]
                pred_win = pred_vals[start:end]
                # print(start, end)


                # Test if an anomalous seq at least equal to the threshold is detected
                count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in
                                        groupby(pred_win) if val==self.pos_val]
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
