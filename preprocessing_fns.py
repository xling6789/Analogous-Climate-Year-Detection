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
from api.client.gro_client import GroClient
from win_iter import smooth_windows

def combine_series(L, names, to_file = None):
    '''Combines a list of preprocessed data series.
    
    Inputs::
        L (list of DataFrames) -- list of dataframes with columns day, yr1, yr2, etc.
                                  NOTE: dataframes must have the same sets of years
        names (list of features) -- list of features corresponding to dataframes
                                    in L
        to_file (string) -- (default None) string of file to write to
    Outputs::
        DataFrame with columns feature, day, yr1, yr2, etc.
    '''
    ret_df = pd.concat([l for l in L], axis=0, ignore_index=True)
    feature_list = []
    for i in range(len(L)):
        feature_list = feature_list + [names[i]]*L[i].shape[0]
    ret_df['feature'] = feature_list
    # reorder columns to fit our convention
    ret_df = ret_df[[list(ret_df.columns)[-1]] + list(ret_df.columns)[:-1]]

    if to_file:
        ret_df.to_csv(to_file)
    else:
        return ret_df

def preprocess(df, to_file = None,
               years = False, leapyear = None,
               standardization = None, handle_na = None,
               start_day = 1, end_day = 366):
    '''preprocess the dataframe so that rows represent Julian days and columns 
    represent observations in each year
    
    Inputs::
        df (DataFrame) -- a GRO data series
        to_file (str) -- (default None) string representing file to write output to
        years (list of ints) -- (default False) if given, these are the years to return
        leapyear (int) -- (default None) Julian day to remove in leap years
        standardization (str) -- (default None) can be 
                                    ``zscore`` - mean 0, std 1
                                    ``minmax`` - min 0, max 1
                                 will be standardized across all years
        handle_na (str) -- (default None) how to handle missing data. Can be
                              ``interpolate`` - interpolates NaNs
                                  NOTE: will not currently interpolate between
                                        years, and will interpolate for day 366.
        start_day(int) -- start day
        end_day(int) -- ending day
    '''
    # some dates are strings...
    df['end_date'] = pd.to_datetime(df['end_date'])
    # add columns for year, Julian day
    df['year'] = [d.year for d in df['end_date']]
    #df['month'] = [d.month for d in df['end_date']]
    df['day'] = [d.dayofyear for d in df['end_date']]
  
    # build the dataframe to return
    # there may be a slight problem with leap years... 
    # for now, we'll just put a zero for the last day
    
    if not years:
        years = np.unique(df['year'])
    
    if leapyear:
        A = np.zeros((365,len(years)))
    else:
        A = np.zeros((366,len(years)))
    A[:] = np.nan
    
    for j in range(len(years)):
        year = years[j]
        #if leapyear:
        #    A = np.zeros(365)
        #else:
        #    A = np.zeros(366)

        #A[:] = np.nan
        yr_df = df[df['year'] == year]
        for i in range(yr_df.shape[0]):
            row = yr_df.iloc[i]
            # checks if leap year
            if leapyear and not year%4:
                # fill in normally here
                if row['day'] < leapyear:
                    A[row['day']-1,j] = row['value']
                elif row['day'] > leapyear:
                    A[row['day']-2,j] = row['value']
            else:
                A[row['day']-1,j] = row['value']
        
    if standardization == 'zscore':
        m = np.mean(A[np.logical_not(np.isnan(A))])
        s = np.std(A[np.logical_not(np.isnan(A))])
        A = A/s - m/s
        
    elif standardization == 'minmax':
        s = np.min(A[np.logical_not(np.isnan(A))])
        l = np.max(A[np.logical_not(np.isnan(A))])
        A = (A - s)/(l - s)
        
    # build the dataframe
    if leapyear:
        ret_df = pd.DataFrame({'day':range(1,366)})
    else:
        ret_df = pd.DataFrame({'day':range(1,367)})

    for i in range(len(years)):
        ret_df[years[i]] = A[:,i]
        
    # fill in missing values
    if handle_na == 'interpolate':
        ret_df = ret_df.interpolate(limit_direction='both')

    ret_df = ret_df[ret_df['day'].between(start_day, end_day) ]
    if to_file:
        ret_df.to_csv(to_file)
    else:
        return ret_df

def get_gro_data(gro_token = None, item = None, 
            region = None, source = None, 
            similarity = 'Euclidean',
            host = 'api.gro-intelligence.com',
            frequency = 1, metric = None, ret_args = False, **kwargs):
    '''
    returns GRO data series as a DataFrame
    
    Inputs::
        gro_token (string) -- token for GRO account
        item (string or int) -- feature or list of features to compare
        region (string or int) -- region to compare
        source (string or int) -- source of data on ``item`` in ``region``
        frequency (string or int) -- (default 1) frequency of observations
        metric (string or int) -- how to measure ``item``
        
    Outputs::
        DataFrame
    '''
    # initialize client
    client = GroClient(host, gro_token)

    keys = ['item_id', 'metric_id', 'region_id', 'frequency_id', 'source_id']
    var_list = [item, metric, region, frequency, source]
    search_names = ['items', 'metrics', 'regions', 'frequencies', 'sources']
    
    bad_input = False
    arg_dict = {}
    
    for i in range(len(keys)):
        if var_list[i]:
            key = keys[i]
            var = var_list[i]
            search = search_names[i]
            
            if isinstance(var, str):
                arg_dict[key] = client.search_for_entity(search, var)
            elif isinstance(var, int):
                arg_dict[key] = var
            else:
                raise ValueError('GRO inputs must be strings or ints,' +
                                 ' but input ' + str(var) + ' is type ' +
                                 str(type(var)))

    # if we were not specific enough, just get the first data series
    arg_dict = client.get_data_series(**arg_dict)[0]
    
    # get data from gro
    if ret_args:
        return pd.DataFrame(client.get_data_points(**arg_dict)), arg_dict
    return pd.DataFrame(client.get_data_points(**arg_dict))

def test():
    gro_token = open('GROAPI_TOKEN_BEN.txt','r').read().strip()
    args_T = {'item': 'Land Temperature', 'region': 'Nanjing'}
    df_T = get_gro_data(gro_token, **args_T)
    processed_T = preprocess(df_T, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366)
    
    args_R = {'item': 'Rainfall (modeled)', 'region': 'Nanjing'}
    df_R = get_gro_data(gro_token, **args_R)
    processed_R = preprocess(df_R, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366)

    args_N = {'item': 'Vegetation (NDVI)', 'region': 'Nanjing',
              'frequency':3}
    df_N = get_gro_data(gro_token, **args_N)
    processed_N = preprocess(df_N, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366)
    combine_series([processed_T, processed_R, processed_N],
                         ['Temperature', 'Rainfall', 'NDVI'], 
                         to_file = 'nanjing_normalized.csv')
    
    
    args_T = {'item': 'Land Temperature', 'region': 'Nanjing'}
    df_T = get_gro_data(gro_token, **args_T)
    processed_T = preprocess(df_T, handle_na = 'interpolate', leapyear=366)
    
    args_R = {'item': 'Rainfall (modeled)', 'region': 'Nanjing'}
    df_R = get_gro_data(gro_token, **args_R)
    processed_R = preprocess(df_R, handle_na = 'interpolate', leapyear=366)

    args_N = {'item': 'Vegetation (NDVI)', 'region': 'Nanjing',
              'frequency':3}
    df_N = get_gro_data(gro_token, **args_N)
    processed_N = preprocess(df_N, handle_na = 'interpolate', leapyear=366)
    combine_series([processed_T, processed_R, processed_N],
                         ['Temperature', 'Rainfall', 'NDVI'], 
                         to_file = 'nanjing.csv')
    
    args_T = {'item': 'Land Temperature', 'region': 'Nanjing'}
    df_T = get_gro_data(gro_token, **args_T)
    processed_T = smooth_windows(preprocess(df_T, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366))
    
    args_R = {'item': 'Rainfall (modeled)', 'region': 'Nanjing'}
    df_R = get_gro_data(gro_token, **args_R)
    processed_R = smooth_windows(preprocess(df_R, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366))

    args_N = {'item': 'Vegetation (NDVI)', 'region': 'Nanjing',
              'frequency':3}
    df_N = get_gro_data(gro_token, **args_N)
    processed_N = smooth_windows(preprocess(df_N, handle_na = 'interpolate',
                           standardization = 'zscore', leapyear=366))
    combine_series([processed_T, processed_R, processed_N],
                         ['Temperature', 'Rainfall', 'NDVI'], 
                         to_file = 'nanjing_normalized_smooth.csv')
    
    
    args_T = {'item': 'Land Temperature', 'region': 'Nanjing'}
    df_T = get_gro_data(gro_token, **args_T)
    processed_T = smooth_windows(preprocess(df_T, handle_na = 'interpolate', 
                                            leapyear=366))
    
    args_R = {'item': 'Rainfall (modeled)', 'region': 'Nanjing'}
    df_R = get_gro_data(gro_token, **args_R)
    processed_R = smooth_windows(preprocess(df_R, handle_na = 'interpolate', 
                                            leapyear=366))

    args_N = {'item': 'Vegetation (NDVI)', 'region': 'Nanjing',
              'frequency':3}
    df_N = get_gro_data(gro_token, **args_N)
    processed_N = smooth_windows(preprocess(df_N, handle_na = 'interpolate', 
                                            leapyear=366))
    combine_series([processed_T, processed_R, processed_N],
                         ['Temperature', 'Rainfall', 'NDVI'], 
                         to_file = 'nanjing_smooth.csv')
    
if __name__=='__main__':
    test()