#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering utility functions for protein sequence analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####################################################
#### Functions for protein sequence descriptors ####
####################################################

def assign_descriptor(sequence, descriptor='vhse'):
    """
    Assign physicochemical descriptors to a protein sequence.
    
    Parameters:
    -----------
    sequence : str
        Protein sequence to convert
    descriptor : str
        Type of descriptor to use: 'vhse', 'zscales', or 'PCA_embeddings'
    
    Returns:
    --------
    list
        Descriptor values for the sequence
    """
    vhse_tbl = [['A', 0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48], 
                ['R', -1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83], 
                ['N', -0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80], 
                ['D', -1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56], 
                ['C', 0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19], 
                ['Q', -0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41], 
                ['E', -1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.02], 
                ['G', -0.20, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
                ['H', -0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65], 
                ['I', 1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13], 
                ['L', 1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62], 
                ['K', -1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13], 
                ['M', 1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68], 
                ['F', 1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.20], 
                ['P', 0.22, -0.17, -0.50, 0.05, -0.01, -1.34, -0.19, 3.56],
                ['S', -0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
                ['T', -0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
                ['W', 1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
                ['Y', 0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52], 
                ['V', 0.76, -0.92, -0.17, -1.91, 0.22, -1.40, -0.24, -0.03]]
    
    zscales_tbl = [['A', 0.24, -2.32, 0.60, -0.14, 1.30],
                   ['R', 3.52, 2.50, -3.50, 1.99, -0.17],
                   ['N', 3.05, 1.62, 1.04, -1.15, 1.61],
                   ['D', 3.98, 0.93, 1.93, -2.46, 0.75], 
                   ['C', 0.84, -1.67, 3.71, 0.18, -2.65],
                   ['Q', 1.75, 0.50, -1.44, -1.34, 0.66],
                   ['E', 3.11, 0.26, -0.11, -3.04, -0.25],
                   ['G', 2.05, -4.06, 0.36, -0.82, -0.38],
                   ['H', 2.47, 1.95, 0.26, 3.90, 0.09],
                   ['I', -3.89, -1.73, -1.71, -0.84, 0.26],
                   ['L', -4.28, -1.30, -1.49, -0.72, 0.84],
                   ['K', 2.29, 0.89, -2.49, 1.49, 0.31], 
                   ['M', -2.85, -0.22, 0.47, 1.94, -0.98], 
                   ['F', -4.22, 1.94, 1.06, 0.54, -0.62],
                   ['P', -1.66, 0.27, 1.84, 0.70, 2.00],
                   ['S', 2.39, -1.07, 1.15, -1.39, 0.67],
                   ['T', 0.75, -2.18, -1.12, -1.46, -0.40],
                   ['W', -4.36, 3.94, 0.59, 3.44, -1.59],
                   ['Y', -2.54, 2.44, 0.43, 0.04, -1.47],
                   ['V', -2.59, -2.64, -1.54, -0.85, -0.02]]
    
    PCA_embeddings_tbl = [['A', -1.71, 2.28, -2.21, -2.08, -2.04, 0.27, 0.20, -0.23, -1.82, 0.87],
                      ['C', -1.86, -0.73, -2.22, -1.86, 0.58, -0.30, 2.42, 1.16, -2.20, -2.62],
                      ['D', -4.33, -2.08, -3.37, 5.38, -0.38, -0.60, -1.16, 0.82, 1.41, 0.96],
                      ['E', -4.06, -0.59, 1.50, 3.15, -3.67, -1.01, -1.30, -2.84, -1.45, 0.19],
                      ['F', -5.36, -1.68, -1.34, 0.98, -0.38, 0.93, -1.10, 0.04, -1.98, 2.80],
                      ['G', -2.19, -0.35, -3.57, -1.25, -3.45, 0.48, 3.73, 1.69, 0.75, -1.02],
                      ['H', -0.00, -3.90, 0.79, 0.28, 3.73, 2.01, -0.05, -1.04, -1.62, 2.81],
                      ['I', -2.53, 5.05, 0.58, 3.07, 1.71, 0.81, 2.01, -1.00, 1.86, -0.51],
                      ['K', -2.41, -0.63, 4.76, -1.22, 0.68, -0.79, 2.00, -0.05, 0.99, 3.30],
                      ['L', -2.71, 2.06, 2.06, 1.56, -1.45, 1.45, -1.99, 4.28, -1.13, 1.67],
                      ['M', -2.46, 2.66, 2.82, 0.41, -0.99, -3.81, -1.53, 3.22, 0.53, -2.70],
                      ['N', -2.59, -1.95, -1.59, 1.66, 4.55, -1.20, 0.28, 2.75, 1.66, -0.60],
                      ['P', -1.11, 0.92, -0.22, -2.49, -0.98, 5.91, -3.41, 0.06, 3.46, -1.35],
                      ['Q', -2.43, -1.29, 4.39, -0.32, -0.64, 0.13, -1.13, -0.98, -2.86, -2.96],
                      ['R', -1.45, -1.48, 3.96, -2.30, 0.93, 0.98, 2.82, 0.70, 0.90, 2.41],
                      ['S', -1.64, 0.74, -3.26, -3.88, 0.41, -1.41, -1.26, 0.67, -1.93, 1.51],
                      ['T', -1.95, 4.10, -1.28, -2.67, 3.25, -2.81, -2.90, -2.75, 0.89, 0.73],
                      ['V', -1.37, 4.86, -0.59, 2.20, 0.07, 1.23, 2.86, -2.75, -0.68, -0.53],
                      ['W', -4.39, -4.19, 0.04, -1.57, -3.02, -2.74, 0.10, -2.15, 4.04, -1.04],
                      ['Y', -4.31, -3.76, -1.24, 0.95, 1.11, 0.49, -0.61, -1.61, -0.84, 1.67]]
    
    seq = [x for x in sequence]
    result = []
    desc_tbl = zscales_tbl if descriptor == 'zscales' else PCA_embeddings_tbl if descriptor == "PCA_embeddings" else vhse_tbl
    for i in seq:
        for j in range(len(desc_tbl)):
            if i == desc_tbl[j][0]:
                for q in desc_tbl[j][1:]:
                    result.append(q)
    return result

def get_descriptors(data_list, descriptor='vhse'):
    """
    Calculate descriptors for a list of protein sequences.
    
    Parameters:
    -----------
    data_list : list
        List of protein sequences
    descriptor : str
        Type of descriptor to use: 'vhse', 'zscales', or 'PCA_embeddings'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing descriptor values for all sequences
    """
    calculated_descriptors = []
    for i in range(len(data_list)):
        seq = data_list[i]
        list_descriptors = assign_descriptor(seq, descriptor)
        calculated_descriptors.append(list_descriptors)
    calculated_descriptors = pd.DataFrame(calculated_descriptors)
    return calculated_descriptors

def norm_features(data):
    """
    Normalize features using MinMaxScaler.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing features to normalize
    
    Returns:
    --------
    pandas.DataFrame
        Normalized DataFrame
    """
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    df_norm = pd.DataFrame(x_scaled)
    return df_norm

####################################################
#### Functions for clustering analysis ####
####################################################

def kMeansRes(scaled_data, k, alpha_k=0.02):
    """
    Calculate scaled inertia for KMeans clustering.
    
    Parameters:
    -----------
    scaled_data : matrix
        Scaled data. Rows are samples and columns are features for clustering
    k : int
        Current k for applying KMeans
    alpha_k : float
        Manually tuned factor that gives penalty to the number of clusters
    
    Returns:
    --------
    float
        Scaled inertia value for current k
    """
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeans(X, k_range):
    """
    Choose the best k value for KMeans clustering based on scaled inertia.
    
    Parameters:
    -----------
    X : matrix
        Data matrix
    k_range : range
        Range of k values to test
    
    Returns:
    --------
    int
        Best k value
    """
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(X, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k

def cluster_analysis(data, num_dimensions):
    """
    Perform t-SNE and KMeans clustering analysis with different perplexity values.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to analyze
    num_dimensions : int
        Number of dimensions for t-SNE
    
    Returns:
    --------
    pandas.DataFrame
        Results of clustering analysis for different perplexity values
    """
    results = []
    perplexity_range = range(5, 51, 5)  # From 5 to 50, step 5
    for perplexity in perplexity_range:
        tsne = TSNE(n_components=num_dimensions, perplexity=perplexity, random_state=42)
        X = tsne.fit_transform(data)
        k_range = range(2, 11)
        
        best_k = chooseBestKforKMeans(X, k_range)
        
        model = KMeans(n_clusters=best_k)
        model.fit(X)
        labels = model.labels_
        silhouette_avg = silhouette_score(X, labels)
        
        # Store results for this perplexity configuration and best k
        results.append({
            'perplexity': perplexity,
            'best_k': best_k,
            'silhouette_score': silhouette_avg
        })
    results = pd.DataFrame(results)
    
    return results

def run_tsne_and_kmeans(data, perplexity, k, num_components):
    """
    Run t-SNE and KMeans with specific parameters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    perplexity : int
        Perplexity parameter for t-SNE
    k : int
        Number of clusters for KMeans
    num_components : int
        Number of components for t-SNE
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with t-SNE coordinates and cluster labels
    """
    # Run t-SNE
    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=42)
    tsne_data = tsne.fit_transform(data)
    
    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tsne_data)
    labels = kmeans.labels_
    
    # Create DataFrame with t-SNE transformed data and KMeans labels
    columns = ['tsne_' + str(i) for i in range(1, num_components + 1)]
    result_df = pd.DataFrame(np.column_stack([tsne_data, labels]), columns=columns + ['label'])
    
    return result_df

def plot_clusters_2d(result_df, title="2D t-SNE with KMeans Clustering", prefix="default"):
    """
    Plot 2D clusters from t-SNE and KMeans results.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with t-SNE coordinates and cluster labels
    title : str
        Plot title
    prefix : str
        Prefix for the output filename (default: "default")
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(result_df['tsne_1'], result_df['tsne_2'], 
                         c=result_df['label'], cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(f'{prefix}_clusters_2d.png', dpi=300)
    plt.show()

def plot_clusters_3d(result_df, title="3D t-SNE with KMeans Clustering", prefix="default"):
    """
    Plot 3D clusters from t-SNE and KMeans results.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with t-SNE coordinates and cluster labels
    title : str
        Plot title
    prefix : str
        Prefix for the output filename (default: "default")
    """
    if 'tsne_3' not in result_df.columns:
        print("Error: DataFrame does not have 3D t-SNE components")
        return
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(result_df['tsne_1'], result_df['tsne_2'], result_df['tsne_3'],
                        c=result_df['label'], cmap='viridis',
                        alpha=0.7, s=50, edgecolors='w')
                        
    plt.colorbar(scatter, label='Cluster')
    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.tight_layout()
    plt.savefig(f'{prefix}_clusters_3d.png', dpi=300)
    plt.show()
