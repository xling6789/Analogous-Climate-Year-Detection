# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:18:19 2019

Authors::
    Xing Ling, Ben Strasser, Rahim Taghikhani, Tianyu Tao, Yiqing Cai

Others have the right to freely use this code in both commercial and
non-commercial applications. However, we retain the right to use this code or
parts of this code for any purpose.
"""

from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

def plot_side(X, special_years, method, legend=True, **kwpar):
    '''
    Input: 
    X: the dataframe to be ploted, presumbably a 365*20 dataframe
    
    special years: list of year to be highlighted, presumbably contains only 2 years.
    
    method {"PCA","TSNE","ISOMAP"}: method to use for dimensionality reduction.
    
    **kwpar: parameters used to control figure size, and set up axis labels, etc.
    
    plots on the left the line chart, on the right the 2d plot of the original data after the dimensionality
    reduction

    Example:
        data_to_plot = smooth_temp_df
        parameters = {'x_label':"weeks",'y_label':"temperature ($^\circ$C)",'fig_size':(16,8)}
        plot_side(data_to_plot,special_year=[2010,2014], method='PCA',legend=True,**parameters)


    
    '''
    # set labels from keyword parameter, probably not the best practice using try except...
    try:
        x_label=kwpar['x_label']
        y_label=kwpar['y_label']
        fig_size=kwpar['fig_size']
    except KeyError:
        x_label = 'x_label'
        y_label = 'y_label'
        fig_size = (12,5)
    fig, ax  = plt.subplots(1,2,figsize=fig_size)

    #sp_color are colors for special years, I have to make two copy of them for each plot.
    X = X.reset_index(drop=True)
    years = X.columns[2:]
    sp_color= iter(["black", "red"])
    
    #plotting the line chart for time series.
    for year in years:
        if year in special_years:
            ax[0].plot(X[year], label=year, color=next(sp_color),linewidth = 2)
        else:
            ax[0].plot(X[year], label='_nolegend_', color='lightgrey', linestyle='dashed',alpha=0.3,)

    ax[0].set_xlabel(x_label, fontsize=14)
    ax[0].set_ylabel(y_label, fontsize=14)
    if legend:
        ax[0].legend()

    #plotting 2d visualization using preferred dimensionally reduction methods, probably should add titles for other use case, but for now I know I only generate those
    #three charts...
    if method == 'PCA':
        plot_method = PCA(n_components=2)
        fig.suptitle('Weekly Temperature Comparison Using Euclidean Distance and PCA', fontsize=25)

    elif method == 'TSNE':
        plot_method = TSNE(n_components=2,perplexity=15)
        fig.suptitle('Weekly Rainfall Comparison Using R-squared and TSNE', fontsize=25)

    elif method == 'ISOMAP':
        plot_method = Isomap(n_neighbors = 3, n_components= 2)
        fig.suptitle('NDVI Comparison Using $L^{\infty}$ Distance and ISOMAP', fontsize=25)

    
    Y = plot_method.fit_transform(X[years].T)
    sp_color_2= iter(["black", "red"])
    ax[1].scatter(Y[:, 0], Y[:, 1])
    for i,year in enumerate(years):
        ax[1].annotate(year, (Y[i, 0], Y[i, 1]))
        if year in special_years:
            ax[1].plot(Y[i, 0], Y[i, 1], 'o', markersize=16, mfc='none', mew=2,mec=next(sp_color_2))
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_compare(X, Y, special_year, legend=True, **titles):
    '''
    Input: 
    X: one dataframe to be ploted, presumbably a 365*20 dataframe.
    X: other dataframe to be ploted, presumbably a 365*20 dataframe.
    
    special year: list of year to be highlighted, usually contain only 2 years.
    
    
    **titles: keyword arguments for the line chart x,y-axis labels.


    NOTE: This function plots on the left the line chart for the time series, on the right the 2d plot of the original data after the dimensionality
    reduction.

    example:
        data_to_plot = combined_df.loc[combined_df['feature']=='Vegetation']
        plot_side(data_to_plot,special_year=[2001,2002], x_title="days",y_title="NDVI")
    
    '''
    try:
        T1 = titles['T1']
        T2 = titles['T2']
        x_label=titles['x_label']
        y_label=titles['y_label']
        fig_size = titles['fig_size']
    except KeyError:
        T1 = 'T1'
        T2 = 'T2'
        x_label = 'x_label'
        y_label = 'y_label'
        fig_size = (12,5)
    fig, ax  = plt.subplots(1,2,figsize=fig_size)

    
    X = X.reset_index(drop=True)
    years = X.columns[2:]
    sp_color= iter(["black", "red"])
    
    #plotting the line chart for time series.
    for year in years:
        if year in special_year:
            ax[0].plot(X[year], label=year, color=next(sp_color),linewidth = 2)
        else:
            ax[0].plot(X[year], label='_nolegend_', color='lightgrey', linestyle='dashed',alpha=0.3,)

    ax[0].set_xlabel(x_label, fontsize=14)
    ax[0].set_ylabel(y_label, fontsize=14)
    
    sp_color= iter(["black", "red"])
    #plotting the line chart for time series.
    for year in years:
        if year in special_year:
            ax[1].plot(Y[year], label=year, color=next(sp_color),linewidth = 2)
        else:
            ax[1].plot(Y[year], label='_nolegend_', color='lightgrey', linestyle='dashed',alpha=0.3,)

    ax[1].set_xlabel(x_label, fontsize=14)
    ax[1].set_ylabel(y_label, fontsize=14)
    if legend:
        ax[1].legend()
        ax[0].legend()
    ax[0].set_title(T1)
    ax[1].set_title(T2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
def plot_similar_yrs(df, year=2019, n=5, shift=False, feat=None, similarity='Euclidean'):
    '''plots years similar to ``year``
    '''
    L = get_sorted_years(year, df, similarity=similarity, shift=shift)[:n]
    for yr in L:
        plt.plot(df[['day']],df[yr[1]], label=yr[1])
    plt.xlabel('day')
    plt.ylabel(feat)
    if not shift:
        plt.title('Similar Years - ' + similarity + ' similarity ')
    else:
        plt.title('Similar Years - shifted ' + similarity + ' similarity ')
    plt.legend()
    plt.show()