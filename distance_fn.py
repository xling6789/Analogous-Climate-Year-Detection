# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:18:19 2019

Authors::
    Xing Ling, Ben Strasser, Rahim Taghikhani, Tianyu Tao, Yiqing Cai

Others have the right to freely use this code in both commercial and
non-commercial applications. However, we retain the right to use this code or
parts of this code for any purpose.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def dist( t1, t2, shift=False, metric='Euclidean',handle_na='interpolate'):
    '''
    Parameters
    ----------

    t1,t2: 
        two time series data to be compared, assuming they are formatted as
        pandas dataframe.
        
    shift (bool):
        determines whether we perform (left and right) shift on the time series data. If yes, find the minmum distance
        between time series for shift values between 1 and max_shift=7 (days). The shift is periodic, for example,
        [0,0,0,1] will be shifted to [1,0,0,0] if we want to shift right by 1 index.
    
    metric {'R2','Euclidean', 'L1', or 'Linf'}: 
        the metric used to compare those two time series, 
        by default the Euclidean distance is used.
    
    handle_na {'interpolate', 'fill', 'drop'}
        the way missing values are handled, 'interpolat' interpolates the missing value based on neighbouring values.
        'fill' just fill the n/a values with the total average. And 'drop' imputes the n/a values when computing the
        distance. The default method is interpolate.

   
    Returns
    ----------
    
    min_shift_dist(float): the distance between the two (shifted) time series using the selected metric.
    
    '''
    if handle_na == 'interpolate':
        t1 = t1.interpolate().fillna(method='ffill').fillna(method='bfill')
        t2 = t2.interpolate().fillna(method='ffill').fillna(method='bfill')
    elif handle_na == 'fill':
        t1 = t1.fillna(t1.mean())
        t2 = t2.fillna(t1.mean())
    elif handle_na == 'drop':
        pass
    else:
        raise ValueError("handle_na should be one of 'interpolate', 'fill','Linf'")

    
    t1 = t1.values
    t2 = t2.values
    
    if shift:
        max_shift = 7
    else:
        max_shift = 0
        
    min_shift_dist = np.inf
    #Consider both left and right shift, so the metric is symmetric
    for shift in range(-max_shift,max_shift+1):
        t2_shifted = np.roll(t2, shift)
        
        if metric == 'Euclidean':
            if handle_na == 'interpolate' or handle_na == 'fill':
                dist = np.linalg.norm(t1-t2_shifted)
            else:
                dist = np.sqrt(np.nansum((t1 - t2_shifted)**2))
            
        elif metric == 'L1':
            if handle_na == 'interpolate' or handle_na == 'fill':
                dist = np.linalg.norm(t1-t2_shifted,1)
            else:
                dist = np.nansum(np.abs(t1 - t2_shifted))
            
        elif metric == 'Linf':
            if handle_na == 'interpolate' or handle_na == 'fill':
                dist = np.linalg.norm(t1-t2_shifted, np.inf)
            else:
                dist = np.nanmax(np.abs(t1-t2_shifted))
        
        elif metric == 'R2':
            if handle_na == 'interpolate' or handle_na == 'fill':
                dist = 1-r2_score(t1,t2_shifted)
            else:
                mask = ~np.isnan(t1)&~np.isnan(t2_shifted)
                dist = 1-r2_score(t1[mask],t2_shifted[mask])
        else:
            raise ValueError("metric should be one of 'Euclidean', 'L1','Linf' or 'R2'.")
        
        if dist <= min_shift_dist:
            min_shift_dist = dist
        
    return min_shift_dist
