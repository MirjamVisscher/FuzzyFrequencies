#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:37:57 2024

@author: mirjam
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import mannwhitneyu, kruskal
import scikit_posthocs as sp
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns



def mann_whitney_test(distances1, distances2):
    u_statistic, p_value = mannwhitneyu(distances1, distances2, alternative='less')
    
    # Calculate effect size
    n1 = len(distances1)
    n2 = len(distances2)
    effect_size = u_statistic / (n1 * n2)
    
    # Calculate Cohen's d
    mean1, mean2 = np.mean(distances1), np.mean(distances2)
    std1, std2 = np.std(distances1, ddof=1), np.std(distances2, ddof=1)
    pooled_std = sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return p_value, effect_size, cohens_d

def get_distances(experiment, profile_type, distance_type):
    """
    Read the distances between the profiles between the pcp or pp of all chromatypes 
    and the symbolic, where the symbolic is the ground truth.

    Parameters:
    experiment (str): The name of the experiment, used to construct file paths.
    profile_type (str): The type of profile, used to construct file paths. 'pp', 'pcp'
    distance_type: 'euclidean', 'squared', 'manhattan'
    Returns:
    pd.DataFrame: distances
    """
    idir = os.path.join('..', 'results', 'output', experiment, 'distances')
    all_distances = pd.read_csv(os.path.join(idir, f'{distance_type} distances_{profile_type}.csv'), index_col=0).drop(columns=['composition'])
    return all_distances


def kruskal_wallis_test(experiment, profile_type,distance_type):
    """
    Perform the Kruskal-Wallis H test for independent samples.

    Parameters:
    *groups : variable number of lists or arrays representing independent samples.

    Returns:
    p_value : The p-value of the test.
    H_statistic : Kruskal-Wallis H statistic.
    """
    
    output_dir = os.path.join('..', 'results', 'output', experiment, 'statistics')
    os.makedirs(output_dir, exist_ok=True)
    
    distances = get_distances(experiment, profile_type,distance_type)
    # all_distances=all_distances.iloc[:500]
    
    groups = [distances[col].dropna().values for col in distances.columns]
    # Perform Kruskal-Wallis test
    H_statistic, p_value = kruskal(*groups)
    
    # Calculate effect size (eta squared)
    n_total = sum(len(group) for group in groups)
    H_max = (n_total - 1)
    eta_squared = H_statistic / H_max
    
    # Create KW-statistics DataFrame
    kw_statistics = pd.DataFrame({
        'H_statistic': [H_statistic],
        'p_value': [p_value],
        'eta_squared': [eta_squared]
    })
    
    output_filename = f'kruskal_wallis_statistics_{distance_type}_{profile_type}.csv'
    output_path = os.path.join(output_dir, output_filename)
    kw_statistics.to_csv(output_path)
    
    # Return the p-value, H statistic, and effect size
    return p_value, H_statistic, eta_squared      

def get_statistics(experiment, profile_type, distance_type):        
    output_dir = os.path.join('..', 'results', 'output', experiment, 'statistics')
    os.makedirs(output_dir, exist_ok=True)
    
    distances = get_distances(experiment, profile_type, distance_type)
    # Calculate mean, median, and standard deviation
    summary = pd.DataFrame({
        'mean': distances.mean(),
        'median': distances.median(),
        'stdev': distances.std()
    })
    
    # Transpose the DataFrame to have column names as row names
    summary = summary
    summary.index.name = 'model'
    output_filename = f'summary_{distance_type}_{profile_type}.csv'
    output_path = os.path.join(output_dir, output_filename)
    summary.to_csv(output_path)
    return summary

def perform_dunn_test(experiment, profile_type, distance_type):
    output_dir = os.path.join('..', 'results', 'output', experiment, 'statistics')
    os.makedirs(output_dir, exist_ok=True)
    # Prepare the data for Dunn's test
    distances = get_distances(experiment, profile_type, distance_type)
    data = [distances [col].dropna().values for col in distances .columns]
    
    # Perform Dunn's test
    dunn_results = sp.posthoc_dunn(data)
    
    # Optionally, you can rename the DataFrame columns and indices to match the original groups
    dunn_results.columns = distances.columns
    dunn_results.index = distances.columns
    
    output_filename = f'dunn_test_{distance_type}_{profile_type}.csv'
    output_path = os.path.join(output_dir, output_filename)
    dunn_results.to_csv(output_path)
    
    return dunn_results

def visualise_distances(experiment, profile_type, plot_type, distance_type, save = False):
    """ Visualise the distributions of the distances between profiles of a chroma type
    and profiles extracted from symbolic representations. The source are distances created by
    Experiment.distances(profile_type, chromatype)
    Args:
        experiment: experiment name
        profile type: pp or pcp
        chroma types: list of chroma types available
        distance_type: 'euclidean', 'squared', 'manhattan'
        plot_type: 'histogram', 'line'
        save if True, the output is saved
    Returns:
        the plot
        """
    output_dir = os.path.join('..', 'results', 'figures', experiment, 'distances')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'distribution_distances_{distance_type}_{plot_type}_{profile_type}.png'
    output_path = os.path.join(output_dir, output_file)
       
    distances = get_distances(experiment, profile_type, distance_type)
    
    mindistance = distances.min().min()
    maxdistance = distances.max().max()
        
    plt.style.use('uu')
    plt.figure(figsize=(10, 4))
    bins = np.linspace(mindistance, maxdistance, 40)   
    
    for column in distances.columns:
        column_data = distances[column].dropna()  # Drop missing values
        
        if plot_type == 'histogram':
            plt.hist(column_data, bins=bins, alpha=0.5, label=column, density=True)
        elif plot_type == 'line':
            sns.kdeplot(column_data, label=column, fill=False, bw_adjust=0.5)
    y_min, y_max = plt.ylim()
    raw_step = max(1, (y_max - y_min) // 10)  # Initial adaptive step
    tick_step = max(5, int(np.ceil(raw_step / 5) * 5))  # Round up to nearest multiple of 5
    
    plt.yticks(np.arange(np.ceil(y_min), np.floor(y_max) + 1, tick_step).astype(int))
    plt.xlim(0,0.4)
    
    plt.xlabel(f'{distance_type.capitalize()} distance to symbolic profile')
    plt.ylabel('compositions')
    # plt.yticks(np.arange(plt.ylim()[0], plt.ylim()[1] + 1, 1).astype(int))

    # plt.title(f'Distribution of Squared Distances for {profile_type}\nUsing Various Chroma Extraction Methods')
    plt.legend(title='extraction method')
    plt.xticks()
        
    if save == True:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")    


