#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:47:27 2024

This code is created to measure the effect of number of voices, year of recording
and instrumentation type on the distance between pitch (class) profiles of
symbolic encodings and multif0 extractions.

@author: mirjam
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_metrics(experiment_object, experiment, chromatype = 'multif0'):

     metrics = experiment_object.metrics
     metrics[['voices', 'year_recording']] = metrics[['voices', 'year_recording']].astype(int)
     metrics['instrumentation_category'] = metrics['instrumentation_category'].astype('category')
     idir = os.path.join('..', 'results','output',experiment, 'distances')
     distances_pp = pd.read_csv(os.path.join(idir, 'euclidean distances_pp.csv'))[['composition', chromatype]]
     distances_pcp = pd.read_csv(os.path.join(idir, 'euclidean distances_pcp.csv'))[['composition', chromatype]]
     metrics = metrics.merge(distances_pp, on='composition')
     metrics = metrics.rename(columns={chromatype: 'pp'})
     metrics = metrics.merge(distances_pcp, on='composition')
     metrics = metrics.rename(columns={chromatype: 'pcp'})
     metrics['decade'] = (metrics['year_recording'] // 10) * 10
     
     return metrics   
    

def plot_exploration(metrics, experiment, chromatype):
    plt.style.use('uu')
    # Define the columns to plot
    columns = ['voices', 'instrumentation_category', 'decade']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes = axes.flatten()

    for ax, column in zip(axes, columns):  # Exclude the last subplot for scatter
        if column == 'instrumentation_category':
            # For categorical data, create a bar chart
            metrics[column].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Instrumentation category')
            ax.set_xlabel('')
            ax.set_ylabel('')
        elif column == 'decade':
            decade_counts = metrics[column].value_counts().sort_index()
            all_decades = pd.Series(0, index=range(metrics[column].min(), metrics[column].max() + 1, 10))
            all_decades.update(decade_counts)
            all_decades.plot(kind='bar', ax=ax)
            ax.set_title('Recording decade')
            ax.set_ylabel('')
        elif column == 'voices':
            metrics[column].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title('Number of voices')
            ax.set_ylabel('Number of compositions')
    
        ax.tick_params(axis='x', rotation=0)
    ax.set_ylabel('')
    plt.tight_layout()
    
    odir = os.path.join('..', 'results', 'figures', experiment, 'performance_effect')
    os.makedirs(odir, exist_ok=True)
    plt.savefig(os.path.join(odir, f'data_exploration_{chromatype}.jpg'))
    plt.show()


def plot_metrics(metrics, experiment, plottype, chromatype):
    """
    Create a grid of plots to visualize the relationship between dependent and independent metrics.

    This function generates a 2x3 grid of plots based on the specified plot type, allowing for 
    visual comparisons of dependent metrics ('pp' and 'pcp') against independent metrics ('voices', 
    'instrumentation_category', and 'decade'). The 'decade' is calculated from the 'year_recording' 
    column.

    Parameters:
    ----------
    metrics : pandas.DataFrame
        A DataFrame containing the metrics data. Must include columns for 'year_recording', 
        'voices', 'instrumentation_category', 'pp', and 'pcp'.
        
    plottype : str
        The type of plot to generate. Options are:
        - 'box': Create box plots.
        - 'violin': Create violin plots.
        - 'scatter': Create scatter plots with a logarithmic scale on the y-axis.

    Returns:
    -------
    None
        Displays the generated plots using matplotlib.

    Raises:
    ------
    ValueError
        If an unsupported plot type is provided.
    
    Notes:
    -----
    - The y-axis for scatter plots is set to a logarithmic scale.
    - The 'decade' is calculated by taking the floor of the 'year_recording' divided by 10 and 
      multiplying by 10 to group years into decades.
    """
    plt.style.use('uu')
    if plottype not in ['box', 'violin', 'scatter']:
        raise ValueError(f"Unsupported plot type: '{plottype}'. Choose from 'box', 'violin', or 'scatter'.")
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    if plottype == 'scatter':
        independent_metrics = ['voices', 'instrumentation_category', 'year_recording']
    else:
        independent_metrics = ['voices', 'instrumentation_category', 'decade']
    dependent_metrics = ['pp', 'pcp']
    
    for i, dep_metric in enumerate(dependent_metrics):
        for j, ind_metric in enumerate(independent_metrics):
            if plottype == 'box':
                sns.boxplot(x=ind_metric, y=dep_metric, data=metrics, ax=axs[i, j])
            if plottype == 'violin':    
                sns.violinplot(x=ind_metric, y=dep_metric, data=metrics, ax=axs[i, j], inner='quartile', scale='area')
                
            if plottype == 'scatter':
                axs[i, j].scatter(metrics[ind_metric], metrics[dep_metric], alpha=0.5, s=2)
                axs[i, j].set_yscale('log')
            axs[i, j].set_xlabel(ind_metric)
            axs[i, j].set_ylabel(dep_metric)
    plt.tight_layout()
    # Save the figure
    odir = os.path.join('..', 'results', 'figures', experiment, 'performance_effect')
    os.makedirs(odir, exist_ok=True)
    plt.savefig(os.path.join(odir, f'{plottype}_performance_effect.jpg'))
    plt.show()
    plt.close()

def regression_analysis(metrics, experiment, chromatype):
    regression_metrics = metrics.drop(columns=['composition'])
    regression_metrics = pd.get_dummies(metrics, columns=['instrumentation_category'])
    regression_metrics = regression_metrics.drop(columns=['instrumentation_category_vocal'])
    regression_metrics = regression_metrics.astype({col: int for col in regression_metrics.select_dtypes(include='bool').columns})
    
    
    # Independent variables
    X = regression_metrics[['voices', 'year_recording'] + [col for col in regression_metrics if col.startswith('instrumentation_category_')]]
    # Dependent variable (for pp)
    y_pp = regression_metrics['pp']
    # Dependent variable (for pcp)
    y_pcp = regression_metrics['pcp']
    
    X = X.apply(pd.to_numeric)
    
    # Fit multiple regression model for pp
    X_pp = sm.add_constant(X)  # Adding a constant term for the intercept
    model_pp = sm.OLS(y_pp, X_pp).fit()  # Fit the model
    pp_summary = model_pp.summary()
    
    # Fit multiple regression model for pcp
    X_pcp = sm.add_constant(X)  # Adding a constant term for the intercept
    model_pcp = sm.OLS(y_pcp, X_pcp).fit()  # Fit the model
    pcp_summary = model_pcp.summary()
    
    # Create output directory
    odir = os.path.join('..', 'results', 'output', experiment, 'performance_effect')
    os.makedirs(odir, exist_ok=True)

    # Save summaries to text files
    with open(os.path.join(odir, f'pp_performance_effect_{chromatype}.txt'), 'w') as f:
        f.write(str(pp_summary))
    
    with open(os.path.join(odir, f'pcp_performance_effect_{chromatype}.txt'), 'w') as f:
        f.write(str(pcp_summary))
    
    return pp_summary, pcp_summary

