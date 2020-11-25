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
import warnings

# something I wrote for the last project. I think this functionality is built
# in somewhere, but I was having a hard time finding exactly what I wanted.
# At any rate, I've built it now, and I think it's pretty fast
class WinSeries:
    """timeseres class to output data in windows.
    """
    def __init__(self,series, win_size = 7, overlap = 6):
        """
        Inputs:
            ``series`` - array-like, with time in first column
            ``win_size`` - size of time window to use
            ``overlap`` - size of overlap of time windows
        """
        # in case we got a DataFrame or list or something
        self.series = np.array(series)
        self.step = win_size
        self.overlap = overlap
        
        self.max_time = self.series[-1,0]
        self.max_ind = self.series.shape[0]-1
        
        self.t = self.series[0,0]
        self.ind = 0
        
    def __iter__(self):
        
        return self
        
    def __next__(self):
        """return the next time window
        """
        if self.t >= self.max_time:
            raise StopIteration 
        end_t = self.t + self.step
        start_ind = self.ind
        next_start = False # indicator if we have found the next start time
        while self.t < end_t:
            if self.t >= end_t - self.overlap and not next_start:
                # values for the next loop
                start_t = self.t
                next_start = self.ind
                
            self.ind += 1
            
            # if we hit the end of the array, just return what we have
            if self.ind > self.max_ind:
                return self.series[start_ind:,:]
            else:
                self.t = self.series[self.ind,0]
        
        ret_A = self.series[start_ind:self.ind,:]
            
        # no overlap case 
        if not next_start:
            next_start = self.ind
            start_t = self.t
        
        self.ind = next_start
        self.t = start_t
        return ret_A

def smooth_windows(df, summary = 'mean', win_len = 7, overlap = 0):
    """returns a new timeseries consisting of windows of size ``win_len``
    overlapping by a factor of ``overlap``
    
    ``summary`` is the summary statistic used - can be mean, min, max, or var
  
    NOTE - there is built in functionality IF you don't have to deal with NaNs...
    NOTE - this can be optimized if needed, just let me know if its too slow!
    """
    # this iterator is quick
    windows = WinSeries(df, win_size=win_len, overlap=overlap)
  
    # this part may be slow on a long timeseries
    # we can optimize it by initializing an array of the
    # appropriate size first and then filling out the array
    ret_rows = []
    
    # hack to avoid a RuntimeWarning I keep getting (even though it doesn't
    # seem accurate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for win in windows:
            # mean ignoring nans
            if summary == 'mean':
                ret_rows.append(np.nanmean(win,axis = 0))
            elif summary == 'min':
                ret_rows.append(np.nanmin(win,axis = 0))
            elif summary == 'max':
                ret_rows.append(np.nanmax(win,axis = 0))
            elif summary == 'var':
                ret_rows.append(np.nanvar(win,axis = 0))
            else:
                raise ValueError('statistic ', str(summary), ' must be `mean`, `var`, `min`, or `max`')
    return pd.DataFrame(ret_rows,columns = df.columns)
  