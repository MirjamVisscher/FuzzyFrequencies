#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:43:44 2024

@author: mirjam
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist

from Experiment import Experiment

# Helper function to set chromatypes
def get_chromatypes(profile_type):
    chromatypes_pcp = ['symbolic', 'multif0', 'mt3', '195f', '214c', 'basicpitch', 'cqt', 'hpcp']
    chromatypes_pp = ['symbolic', 'multif0', 'mt3', '195f', '214c', 'basicpitch']
    return chromatypes_pcp if profile_type == 'pcp' else chromatypes_pp


def get_profiles(experiment, profile_type):
    chromatypes = get_chromatypes(profile_type)
    idir = os.path.join('..', 'results', 'output', experiment, profile_type + '_normalised')
    experiment_object = Experiment(experiment)
    profile_frame = pd.DataFrame({'composition': experiment_object.compositions})

    for chromatype in chromatypes:
        try:
            profiles = []
            for comp in experiment_object.compositions:
                # print(comp)
                profile_data = pd.read_csv(os.path.join(idir, f"{comp}.csv"))[chromatype].values
                profiles.append(profile_data)
            profile_frame[chromatype] = profiles
        except:
            pass
    return profile_frame

def n_clusters(profile_frame, chromatype):
    X = np.array(profile_frame[chromatype].to_list())
    inertia, silhouette_scores = [], []
    range_n_clusters = range(2, 15)

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
        inertia.append(kmeans.inertia_)
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.plot(range_n_clusters, inertia, 'bx-'), plt.title('Elbow Method'), plt.xlabel('Clusters'), plt.ylabel('Inertia')
    plt.subplot(1, 2, 2), plt.plot(range(2, 15), silhouette_scores, 'rx-'), plt.title('Silhouette Score'), plt.xlabel('Clusters'), plt.ylabel('Score')
    plt.tight_layout(), plt.show()
    
    
    optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores)) + 1]+1
    return optimal_n_clusters

def perform_clustering(X, n_clusters):
    """Fit KMeans clustering to the data and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


def reduce_dimensionality(X, method):
    """Perform dimensionality reduction using PCA or t-SNE."""
    if method == 'PCA':
        return PCA(n_components=2).fit_transform(X)
    elif method == 'tSNE':
        return TSNE(n_components=2, random_state=42).fit_transform(X)
    return None


def plot_clusters(X_reduced, profile, method, symbolic_clusters=None):
    """Create a scatter plot of the clusters."""
    plt.figure(figsize=(8, 6))
    scatter_colors = symbolic_clusters if symbolic_clusters is not None else profile['cluster']
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=scatter_colors, cmap='viridis')
    plt.title(f'KMeans Clustering with {method}, {n_clusters} clusters')
    plt.colorbar(scatter)
    plt.show()





def cluster(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatype, symbolic_clusters=None):
    """Cluster the profiles"""
    profile = profile_frame[chromatype].to_list()
    X = np.array(profile)
    
    # Include the composition column
    profile = pd.DataFrame(profile)
    profile['composition'] = profile_frame['composition'].values  # Add composition information

    # Step 2: Perform KMeans clustering
    profile['cluster'] = perform_clustering(X, optimal_n_clusters)

    # Step 3: Dimensionality Reduction
    X_reduced = reduce_dimensionality(X, method)

    # Compute centroids
    centroids = profile.iloc[:, :-2].groupby(profile['cluster']).mean().values  # Exclude 'composition' and 'cluster' columns

    # Find the most central point for each cluster
    central_points = []
    for i, centroid in enumerate(centroids):
        cluster_subset = profile[profile['cluster'] == i]
        cluster_points = cluster_subset.iloc[:, :-2].values  # Exclude 'composition' and 'cluster'
        distances = cdist(cluster_points, centroid.reshape(1, -1), metric='euclidean')
        central_idx = np.argmin(distances)
        central_points.append(cluster_subset.iloc[central_idx])  # Keep all columns including 'composition'

    central_points_df = pd.DataFrame(central_points)  # Convert to DataFrame for easy handling
    odir_clusters = os.path.join('..', 'results', 'output', experiment, 'clusters')
    os.makedirs(odir_clusters, exist_ok=True)
    filename = f'centroids_{profile_type}_{chromatype}.csv'
    central_points_df.to_csv(os.path.join(odir_clusters, filename))
    return profile, profile['cluster'], central_points_df


def create_subplots(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatypes, symbolic_clusters, mirror=None):
    """Create subplots for clusters of multiple chromatypes with optional mirroring transformations."""

    # Define your custom colormap based on the color list from your .mplstyle file
    custom_colors = ['#5B2182', '#FFCD00', '#C00A35', '#24A793', '#5287C6']
    custom_cmap = ListedColormap(custom_colors)

    n_cols = 4 if profile_type == 'pcp' else 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = axes.flatten()

    # If mirror is None, use default (1,1) for all chromatypes (i.e., no mirroring)
    if mirror is None:
        mirror = [(1, 1)] * len(chromatypes)
    elif len(mirror) != len(chromatypes):
        raise ValueError("Length of 'mirror' must match length of 'chromatypes'")

    for i, (chromatype, (mx, my)) in enumerate(zip(chromatypes, mirror)):
        try:
            profile = profile_frame[chromatype].to_list()
            X = np.array(profile)
            profile_frame['cluster'] = perform_clustering(X, optimal_n_clusters + 1)
            X_reduced = reduce_dimensionality(X, method)
    
            # Apply mirroring only if mirror is specified
            X_reduced[:, 0] *= mx  # Mirror over x-axis if mx = -1
            X_reduced[:, 1] *= my  # Mirror over y-axis if my = -1
    
            axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=symbolic_clusters, cmap=custom_cmap, s=10, alpha=0.4)
            axes[i].set_title(chromatype)
        except:
            pass
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Save the figure
    odir = os.path.join('..', 'results', 'figures', experiment, 'clusters')
    os.makedirs(odir, exist_ok=True)
    plt.savefig(os.path.join(odir, f'{profile_type}-clusters_subplots_{method}.jpg'))
    plt.show()


def create_clusters_with_subplots(experiment, profile_type, method, mirror=None):
    """Combine steps to create clusters and generate subplots.
    mirror: specify over which axis the subplots should be mirrored. 1 = no mirror, -1 =  mirror
    """
    profile_frame = get_profiles(experiment, profile_type)
    optimal_n_clusters = n_clusters(profile_frame, 'symbolic')
    print('optimal_n_clusters: '+str(optimal_n_clusters))
    symbolic_profile, symbolic_clusters,_ = cluster(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatype='symbolic')
    chromatypes = get_chromatypes(profile_type)
    create_subplots(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatypes, symbolic_clusters, mirror)


def evaluate_clusters(combined_clusters, chromatypes):
    """Evaluate clusters using ARI and NMI scores."""
    results = []
    symbolic_clusters = combined_clusters['symbolic']

    for chromatype in chromatypes:
        try:
            if chromatype != 'symbolic':
                current_clusters = combined_clusters[chromatype]
                ari = adjusted_rand_score(symbolic_clusters, current_clusters)
                nmi = normalized_mutual_info_score(symbolic_clusters, current_clusters)
                ami = adjusted_mutual_info_score(symbolic_clusters, current_clusters)
                results.append({'Chromatype': chromatype, 'ARI': ari, 'NMI': nmi, 'AMI':ami})
        except:
            pass

    return pd.DataFrame(results)


def create_clusters_with_evaluation(experiment, profile_type, method):
    """Create clusters and evaluate them."""
    profile_frame = get_profiles(experiment, profile_type)
    optimal_n_clusters = n_clusters(profile_frame, 'symbolic')
    symbolic_profile, symbolic_clusters,_ = cluster(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatype='symbolic')
    
    combined_clusters = pd.DataFrame({'composition': profile_frame['composition'], 'symbolic': symbolic_clusters.reset_index(drop=True)})
    chromatypes = get_chromatypes(profile_type)

    for chromatype in chromatypes:
        if chromatype != 'symbolic':
            try:
                _, clusters, _ = cluster(experiment, profile_frame, optimal_n_clusters, profile_type, method, chromatype, symbolic_clusters)
                combined_clusters[chromatype] = clusters.reset_index(drop=True)
            except:
                pass

    evaluation_df = evaluate_clusters(combined_clusters, chromatypes)
    odir = os.path.join('..', 'results', 'output', experiment, 'statistics')
    os.makedirs(odir, exist_ok=True)
    fname = f'cluster_evaluation_{profile_type}.csv'
    evaluation_df.to_csv(os.path.join(odir, fname))
    
    odir_clusters = os.path.join('..', 'results', 'output', experiment, 'clusters')
    os.makedirs(odir_clusters, exist_ok=True)
    filename = f'clusters_{profile_type}.csv'
    combined_clusters.to_csv(os.path.join(odir_clusters, filename))
    
    return combined_clusters, evaluation_df



    
