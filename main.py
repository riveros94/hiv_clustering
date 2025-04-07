#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 01:26:04 2025

@author: rocio
"""

"""
Main script for KMEANS protein sequence clustering analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clustering_utils import (
    get_descriptors, 
    norm_features, 
    cluster_analysis, 
    run_tsne_and_kmeans, 
    plot_clusters_2d,
    plot_clusters_3d
)


# General data
data = pd.read_csv('data/pi_sequences_classification.csv')
data= data.iloc[1:].reset_index(drop=True)

seq = list(data['Sequence']) # For descriptor conversion


# 1. Clustering SEQUENCE data
sequences = pd.read_csv('data/alignment_matrix.csv', header=None)
results = cluster_analysis(sequences, 2)
results.to_csv("seq_cluster_analysis_results.csv", index=False)
    
best_config = results.loc[results['silhouette_score'].idxmax()]
best_perplexity = int(best_config['perplexity'])
best_k = int(best_config['best_k'])
    
result_df = run_tsne_and_kmeans(sequences, best_perplexity, best_k, 2)
data['cluster'] = result_df['label']
data.to_csv('pi_sequence_cluster_classification.csv', index=False)
plot_clusters_2d(result_df, prefix='sequence')


# 2. VHSE clustering
vhse_data = get_descriptors(seq, descriptor='vhse')
vhse_data.to_csv('data/pi_vhse.csv', index=False)
vhse_data_norm = norm_features(vhse_data)
results = cluster_analysis(vhse_data_norm, 2)
results.to_csv("vhse_cluster_analysis_results.csv", index=False)
    
best_config = results.loc[results['silhouette_score'].idxmax()]
best_perplexity = int(best_config['perplexity'])
best_k = int(best_config['best_k'])

result_df = run_tsne_and_kmeans(vhse_data_norm, best_perplexity, best_k, 2)
data['cluster'] = result_df['label']
data.to_csv('pi_vhse_cluster_classification.csv', index=False)
plot_clusters_2d(result_df, prefix='vhse')

# 3. zScales clustering
zscales_data = get_descriptors(seq, descriptor='zscales')
zscales_data.to_csv('data/pi_zscales.csv', index=False)
zscales_data_norm = norm_features(zscales_data)
results = cluster_analysis(zscales_data_norm, 2)
results.to_csv("zscales_cluster_analysis_results.csv", index=False)
    
best_config = results.loc[results['silhouette_score'].idxmax()]
best_perplexity = int(best_config['perplexity'])
best_k = int(best_config['best_k'])

result_df = run_tsne_and_kmeans(zscales_data_norm, best_perplexity, best_k, 2)
data['cluster'] = result_df['label']
data.to_csv('pi_zscales_cluster_classification.csv', index=False)
plot_clusters_2d(result_df, prefix='zscales')

# 4. PCA_embedding clustering
pca_emb_data = get_descriptors(seq, descriptor='PCA_embeddings')
pca_emb_data.to_csv('data/pi_pca_emb.csv', index=False)
pca_emb_data_norm = norm_features(pca_emb_data)
results = cluster_analysis(pca_emb_data_norm, 2)
results.to_csv("pca_emb_cluster_analysis_results.csv", index=False)
    
best_config = results.loc[results['silhouette_score'].idxmax()]
best_perplexity = int(best_config['perplexity'])
best_k = int(best_config['best_k'])

result_df = run_tsne_and_kmeans(pca_emb_data_norm, best_perplexity, best_k, 2)
data['cluster'] = result_df['label']
data.to_csv('pi_pca_emb_cluster_classification.csv', index=False)
plot_clusters_2d(result_df, prefix='pca_emb')
