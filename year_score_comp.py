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
#from fastdtw import fastdtw
#from scipy.spatial.distance import euclidean
from preprocessing_fns import *
from distance_fn import dist
from plot_fns import *

def get_mat(df, start_day = 1, end_day = 365,
            similarity = 'Euclidean',
            smooth = False, dropnasim = True,
            handle_na = 'interpolate', shift = False, **kwargs):
    '''
    returns matrix of similarity scores by year
    
    Inputs::
        df (DataFrame) -- DataFrame of preprocessed GRO data
        start_day (int) -- (default 1) int at start of comparison period
        end_day (int) -- (default 365) int at end of comparison period
        similarity (str) -- (default "Eucludean") similarity measurement to use
        smooth (tuple of ints) -- (default (7,0)) smoothing period, overlap
        dropnasim (boolean) -- (default True) if True, ignores years which
                                begin or end with NaN
        shift (boolean) -- (default False) if true, considers shifted series
        
    Outputs::
        similarity matrix as an ndarray, list of years
    
    '''
    if not isinstance(similarity, str):
        raise ValueError('``similarity`` input must be a string')

    if smooth and not (isinstance(smooth,tuple) or isinstance(smooth,list)):
        raise ValueError('``trans`` input must be a tuple or list of integers')

    # Sometimes the first column contains feature names
    if df.columns[0] == 'feature':
        yr_ind = 2
    else:
        yr_ind = 1
    
    
    # chop off the days that are too early or too late
    df = df[df['day'].between(start_day, end_day)]
    
    if smooth:
        # slight hack here since our window iterator only understands numerical
        # data and it assumes the first column represents times
        df = smooth_windows(df.iloc[:,yr_ind-1:],
                            win_len = smooth[0], overlap = smooth[1])
        
    # if a year begins or ends with a string of NaN, our distance function
    # will return NaN. This function filters that
    years = list(df.columns[yr_ind:])
    if dropnasim:
        for year in years:
            if np.isnan(df[year].iloc[0]) or np.isnan(df[year].iloc[-1]):
                years.remove(year)

    num_years = len(years)
    sim_mat = np.zeros((num_years, num_years))
    for i in range(num_years):
        for j in range(num_years):
            sim_mat[i,j] = dist(df[years[i]], df[years[j]],
                                metric=similarity, handle_na=handle_na, shift=shift)

    return sim_mat, years

def get_sorted_years(year, df, years = None,
                     start_day = 1, end_day = 365,
                     similarity = 'Euclidean',
                     smooth = False, dropnasim = True,
                     handle_na = 'interpolate', shift = False):
    '''get a list of years and distances similar to a given year
    Inputs::
        year (int) -- year to compare other years to
        df (DataFrame) -- DataFrame of preprocessed GRO data
        years (list of ints) -- (default all years in df) years to compare year to
        start_day (int) -- (default 1) int at start of comparison period
        end_day (int) -- (default 365) int at end of comparison period
        similarity (str) -- (default "Eucludean") similarity measurement to use
        smooth (tuple of ints) -- (default None) smoothing period, overlap
        dropnasim (boolean) -- (default True) if True, ignores years which
                                begin or end with NaN
        shift (boolean) -- (default False) if true, considers shifted series
        
    Outputs::
        a list of tuples of the form (similarity_to_given_year, year)
    '''
    if not isinstance(similarity, str):
        raise ValueError('``similarity`` input must be a string')
        
    if smooth and not isinstance(smooth,tuple):
        raise ValueError('``trans`` input must be a tuple of integers')

    # Sometimes the first column contains feature names
    if df.columns[0] == 'feature':
        yr_ind = 2
    else:
        yr_ind = 1
        
    df = df[df['day'].between(start_day, end_day)]
    
    if smooth:
        # slight hack here since our window iterator only understands numerical
        # data and it assumes the first column represents times
        df = smooth_windows(df.iloc[:,yr_ind-1:],
                            win_len = smooth[0], overlap = smooth[1])
    if not years:
        years = list(df.columns[yr_ind:])
        
    if dropnasim:
        for y in years:
            if np.isnan(df[year].iloc[0]) or np.isnan(df[year].iloc[-1]):
                years.remove(year)
    dists = []    
    
    for y in years:
        dists.append((dist(df[year], df[y],
                          metric=similarity, handle_na=handle_na, shift=shift),y))
    dists = sorted(dists)
    
    return dists

def _compute_from_input():
    '''helper function to compute distances from user input
    '''
    parser = argparse.ArgumentParser(description='Compute similarities between years')
    parser.add_argument('--gro_token', default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOjc2NzgsImhvc3RUeXBlIjoiYXBpIiwiaWF0IjoxNTYzOTk0MzMxfQ.10N1nV-ByMQt1muV3yETk53PyEV5Wz9cs9LrWiffMBA',
                        help='token for GRO API')
    parser.add_argument('--host', default='api.gro-intelligence.com',
                        help='GROAPI host')
    # data arguments
    parser.add_argument('--item', help='item to compare') 
    parser.add_argument('--metric', help='metric to measure ``item``')
    parser.add_argument('--region', help='region to compare')
    parser.add_argument('--source', help='source of data')
    parser.add_argument('--frequency', type=int, default=1,
                        help='frequency of observations')
    # distance arguments
    parser.add_argument('--similarity', default='Euclidean', 
                        help='similarity measurement to use')
    parser.add_argument('--year', type=int,
                        help='year to compare other years to')
    parser.add_argument('--years', type=int, nargs='+',
                        help='list of years to consider')
    parser.add_argument('--ndecimals', type=int, default=1,
                        help="number of decimals to round to (can't be 0)")
    parser.add_argument('--start_day', type=int, default=1,
                        help='start day for comparison')
    parser.add_argument('--end_day', type=int, default=365,
                        help='end day for comparison')
    parser.add_argument('--smooth', type=int, nargs=2, default=[7,0],
                        help='smoothing factors: num_days, num_overlap')
    parser.add_argument('--dropnasim', type=bool, default=True,
                        help='what to do when smilarities are NaN (typically occurs with incomplete data for the start or end of a year)')
    parser.add_argument('--handle_na', default='interpolate',
                        help='how to deal with missing data')
    parser.add_argument('--make_plot', type=str,
                        help='``tseries`` to plot timeseries, ``years`` to plot years' )
    parser.add_argument('--verb', type=str, default='v',
                        help='``v`` for updates, ``n`` for no updates ')
    
    args = parser.parse_args()
    
    D = vars(args)
    
    # make sure the years includes year if it exists
    if D['year'] and D['years']:
        if D['year'] not in D['years']:
            D['years'].append(D['year'])
    
    if D['verb'] == 'v': 
        print('\ncommunicating with GRO API')
    gro_data = get_gro_data(**D)
    
    if D['verb'] == 'v': 
        print('preprocessing data')
    gro_data = preprocess(gro_data, handle_na=D['handle_na'])
    if D['verb'] == 'v': 
        print('computing '+ str(D['similarity']) + ' similarities \n')

    mat, yrs = get_mat(gro_data, **D)
    
    if D['ndecimals']:
        mat = np.round(mat, decimals = D['ndecimals'])
    
    # chop off years we don't want
    # in the future, maybe we do this earlier?
    if D['years']:
        for yr in D['years']:
            if yr not in yrs:
                raise ValueError('year ' + str(yr)+ ' not in data series')
        # not sure why we need the double index here, but I was having issues
        # doing this with a single slice
        inds = [yrs.index(yr) for yr in D['years']]
        mat = mat[inds,:][:,inds]
        yrs = sorted(D['years'])
    
    # later, we'll only compute what we need using Rahim's function
    if D['year']:
        dists = []
        if D['year'] not in yrs:
            raise ValueError('year ' + str(D['year'])+ ' not in data series')
        
        for i in range(len(yrs)):
            if yrs[i] != D['year']:
                dists.append((mat[yrs.index(D['year']), i], yrs[i]))
        
        dists = sorted(dists)
        
        print('similarity scores to year: ' + str(D['year']))
        for d in dists:
            print(str(d[1]) + ':   ' + str(d[0]))
    
    else:
        print(str(D['similarity']) + ' similarity matrix:')
        print(mat)
        print('\nyears:')
        print(yrs)
        
    if D['make_plot']:
        
        if D['make_plot'] == 'tseries':
            print("will plot timeseries " + str(yrs))
        if D['make_plot'] == '2d':
            print("will plot 2d " + str(yrs))
    
def test():
    # sample code - end to end script does not currently handle multiple 
    # items well
    gro_token = open('GROAPI_TOKEN_BEN.txt','r').read().strip()
    tmp_args = {'item': 'Land Temperature', 'region': 'Nanjing'}
    rain_args = {'item': 'Rainfall (modeled)', 'region': 'Nanjing'}
    veg_args = {'item': 'Vegetation (NDVI)', 'region': 'Nanjing',
              'frequency':3}
    args_list = [tmp_args, rain_args, veg_args]
    names = ['Temperature','Rainfall','Vegetation']
    compare_year = 2010
    compare_to = [2001, 2002, 2003, 2004, 2005,
                  2006, 2007, 2008, 2009, 2010,
                  2011, 2012, 2013, 2014, 2015,
                  2016, 2017, 2018]
    
    # in case our compare_year is not in compare_to
    model_years = sorted(list(set([compare_year] + compare_to)))
    
    dataseries_list = [get_gro_data(gro_token,**args) for args in args_list]
    processed_list = [preprocess(series, handle_na='interpolate', 
                                     standardization='zscore',
                                     leapyear=366) 
                      for series in dataseries_list]
    combined_df = combine_series(processed_list, names)
    
    for sim in ['R2', 'Euclidean', 'L1', 'Linf']:
        print('Finding years most similar to ' + str(compare_year) + ' using '
              + sim + ' similarity')
        # in the combined df, the first column contains strings, so years
        # don't start till the third 
        # compute to test that we can do it
        mat, mat_years = get_mat(combined_df, similarity=sim)
        
        sorted_years = get_sorted_years(compare_year, combined_df,
                                        years=compare_to, smooth=(7,0),
                                        similarity=sim)
        if sim == 'R2':
            sorted_years.reverse()
        print('year', 'score')
        for year in sorted_years:
            print(year[1], year[0])
        print()
    
if __name__=='__main__':
    import sys
    # check if we got system arguments
    if len(sys.argv) > 1:
        import argparse
        _compute_from_input()
    # otherwise, run our test script
    else:
        test()
    