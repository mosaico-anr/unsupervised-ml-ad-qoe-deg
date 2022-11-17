import time

from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import numpy as np


def get_performance(y_pred, y_true, test_score, y_win_adjust=None, metric_type='pw', window_type='wad'):
    if metric_type == 'pw': # Point wise
        # Reshape ground truths from (n_samples, win_size) -> (n_samples*win_size)
        y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1], -1)
        #y_pred = y_true.reshape(y_pred.shape[0]*y_pred.shape[1], -1)
        #test_score = test_score.reshape(test_score.shape[0]*test_score.shape[1], -1)
        #print(y_true.shape, y_pred.shape, test_score.shape)
        
        P, R, F1, auc, mcc, accuracy_anom, accuracy_norm = performance_evaluations(y_true, y_pred, y_score=test_score, return_values=True,
                                                plot_roc_curve=False, verbose=True)
        return P, R, F1, auc, mcc,  accuracy_anom, accuracy_norm 
    elif metric_type == 'window':
        # Reshape y_pred from (n_samples*win_size) -> (n_samples, win_size)
        y_pred = y_pred.reshape(y_true.shape[0], -1)
        
        # Create the window labels ground-truths/predictions based on the number of consecutive anomalous observations in the window.
        windows_preds = []
        windows_true = []
        #thresh = 0
        
        for i in range(y_true.shape[0]):
            win_pred = y_pred[i]
            win_obj = y_win_adjust[i]
            win_true_adjust = win_obj.get_ground_truths(window_type=window_type)
            win_pred_adjust = win_obj.compute_window_wised_pred(win_pred, window_type=window_type)
            
            windows_true.append(win_true_adjust)
            windows_preds.append(win_pred_adjust)
            
        windows_preds = np.concatenate(windows_preds)
        windows_true = np.concatenate(windows_true)
        
        # Compute the window test score as the mean of the observations test scores of the window
        #print(test_score.shape)
        #test_score_win = np.mean(test_score.reshape(len(windows_preds),-1), axis=1)
        #print(test_score_win.shape)
        
        P, R, F1, auc, mcc, accuracy_anom, accuracy_norm = performance_evaluations(windows_true, windows_preds, y_score=None, return_values=True,
                                                plot_roc_curve=False, verbose=True)
        return P, R, F1, auc, mcc, accuracy_anom, accuracy_norm
        
    else:
        raise ValueError('Metric type unknown ! Must be pw (point-wise) or window !')
        

def get_best_score(test_score, y_true, y_win_true, y_win_adjust, val_ratio, n_pertiles, metric_type='pw', seed=42, window_type='wad'):
    print(test_score.shape, y_true.shape, y_win_true.shape)
    # Take val_ratio % of the predictions to find the threshold that yield to the
    # best F1 score with a stratified sampling
    if metric_type== 'pw':
        # Reshape ground truths from (n_samples, win_size) -> (n_samples*win_size)
        y_true = y_true.reshape(test_score.shape[0], -1)
        
        # Stratified shuffle based on anomalous observations
        split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        for t_index, v_index in split.split(test_score, y_true):
            score_t, y_t = test_score[t_index], y_true[t_index]
            score_v, y_v = test_score[v_index], y_true[v_index]
        win_adj_t = None
        threshold = get_best_f1_threshold(score_v, y_v, y_win_adjust=None, number_pertiles=n_pertiles, metric_type=metric_type)
        

    elif metric_type== 'window':
        # Reshape test score from  (n_samples*win_size) -> (n_samples, win_size)
        test_score = test_score.reshape(y_win_true.shape[0], -1)
        
        
        # Stratified shuffle based on anomalous windows
        split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        for t_index, v_index in split.split(test_score, y_win_true):
            score_t, y_t, y_win_t = test_score[t_index,:], y_true[t_index,:], y_win_true[t_index]
            score_v, y_v, y_win_v = test_score[v_index,:], y_true[v_index,:], y_win_true[v_index]
            win_adj_t = [y_win_adjust[i] for i in t_index]
            win_adj_v = [y_win_adjust[i] for i in v_index]
        
        # score_v = np.mean(score_v, axis=1)
        # score_t = np.mean(score_t, axis=1)
        threshold = get_best_f1_threshold(score_v, y_win_v, y_win_adjust=win_adj_v, number_pertiles=n_pertiles, metric_type=metric_type, window_type=window_type)
        #threshold = 50000
        #print(score_t.shape, y_t.shape, y_win_t.shape, len(win_adj_t))
        
    else:
        raise ValueError('Metric type unknown ! Must be pw (point-wise) or window !')
        
    
    y_pred = (score_t >= threshold).astype(int)
    precision, recall, f1, auc, mcc, accuracy_anom, accuracy_norm = get_performance(y_pred, y_t, score_t, y_win_adjust=win_adj_t, metric_type=metric_type, window_type=window_type)
    return precision, recall, f1, auc, mcc, accuracy_anom, accuracy_norm
    
def get_best_f1_threshold(test_score, y_true, y_win_adjust, number_pertiles, verbose=True, metric_type='pw', window_type='wad'):
    '''
    '''
    ratio = float(100 * sum(y_true == 0) / len(y_true))
    #print(ratio, type(ratio))
    print(f"Ratio of normal data: {ratio:.2f}%")
    q = np.linspace(max(ratio - 5, 0), min(ratio + 5, 100), number_pertiles)
    thresholds = np.percentile(test_score, q)

    f1 = np.zeros(shape=number_pertiles)
    r = np.zeros(shape=number_pertiles)
    p = np.zeros(shape=number_pertiles)
    
    st_tm = time.time()
    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        y_pred = (test_score >= thresh).astype(int)
        if metric_type == 'window':
            # Create the window labels ground-truths/predictions based on the number of consecutive anomalous observations in the window.
            windows_preds = []
            windows_true = []
            #thresh = 0

            for q in range(y_pred.shape[0]):
                win_pred = y_pred[q]
                win_obj = y_win_adjust[q]
                win_true_adjust = win_obj.get_ground_truths(window_type=window_type)
                win_pred_adjust = win_obj.compute_window_wised_pred(win_pred, window_type=window_type)

                windows_true.append(win_true_adjust)
                windows_preds.append(win_pred_adjust)

            windows_preds = np.concatenate(windows_preds)
            windows_true = np.concatenate(windows_true)
            
            tn, fp, fn, tp = confusion_matrix(windows_true, windows_preds).ravel()
        elif metric_type == 'pw':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            raise ValueError('Metric type unknown ! Must be pw (point-wise) or window !')
        
        p[i] = tp/(tp+fp)
        r[i] = tp /(tp+fn)
        f1[i] = 2*p[i]*r[i] / (p[i] + r[i])
    
    print(f'Threshold optimization took : {time.time() - st_tm:.2f} s')
    arm = np.argmax(f1)
    if verbose:
        print(f"Best metrics with threshold = {thresholds[arm]:.2f} are :\tPrecision = {p[arm]:.2f}\tRecall = {r[arm]:.2f}\tF1-Score = {f1[arm]:.2f}\n")
    return thresholds[arm]

def performance_evaluations(y_true, y_pred, y_score=None, return_values=False, plot_roc_curve=True, verbose=True):
    '''
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp /(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    accuracy_anom = tp/(tp+fn)
    accuracy_norm = tn/(tn+fp)
    f1 = 2*precision*recall / (precision + recall)
    mcc = matthews_corrcoef(y_true, y_pred)
    #print(precision, recall)
    
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true,y_score)
        auc_val = roc_auc_score(y_true,y_score, labels=[0,1])
    
        #average_precision = average_precision_score(y_true, y_score)
        #precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    else:
        auc_val = -1

    #print(average_precision)
    
    # Data to plot precision - recall curve
    # Use AUC function to calculate the area under the curve of precision recall curve
    #aupr = auc(recalls, precisions)
    #print(aupr)
    
    if verbose:
        print(f"Performance evaluation :\nPrecision = {precision:.2f}\nRecall = {recall:.2f}\nAccuracy = {accuracy:.2f}\nF1-Score = {f1:.2f}\nMCC = {mcc:.2f}\nAUC = {auc_val:.2f}\n")
        #print(f"AP = {average_precision:.2f}\nAUPR = {aupr:.2f}\n")
    
    if plot_roc_curve:        

        #idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()
        plt.figure()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(fpr,tpr,label=f"AUPR = {auc_val:.2f}", color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        #plt.plot(fpr[idx],tpr[idx], 'ro')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
    if return_values:
        return precision, recall, f1, auc_val, mcc, accuracy_anom, accuracy_norm
