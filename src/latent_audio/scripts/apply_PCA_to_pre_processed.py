"""This script takes the data as output by the pre_process script and applies a principal component analysis to each layer 
to reduce the dimensionality. This is done to allow for equal model sizes within the disentangle script."""

import os
from latent_audio import utilities as utl
from sklearn.decomposition import PCA
import numpy as np

# Configuration
pre_processed_folder = os.path.join('data','pre-processed','All PCA dimensions')
pca_dimensionality = 64
output_folder = os.path.join('data', 'pre-processed',f'{pca_dimensionality} PCA dimensions all in 1 file')

sample_size = 2048

for l in range(14):
    np.random.seed(42)
    layer_folder_pre_processed = os.path.join(pre_processed_folder, f'Layer {l}')
    layer_folder_ouput = os.path.join(output_folder, f'Layer {l}')
    if not os.path.exists(layer_folder_ouput): os.makedirs(layer_folder_ouput)
    
    # Load sample
    X_sample, Y_sample = utl.load_latent_sample(data_folder=layer_folder_pre_processed, sample_size=sample_size)

    # Fit PCA
    pca = PCA(n_components=pca_dimensionality)
    pca.fit(X_sample)

    # Transform each X
    x_file_names = utl.find_matching_strings(strings=os.listdir(layer_folder_pre_processed), token='_X_')
    Xs = [None] * len(x_file_names); Ys = [None] * len(x_file_names)
    for i, x_file_name in enumerate(x_file_names):
        X = np.load(os.path.join(layer_folder_pre_processed, x_file_name))[np.newaxis,:]
        Xs[i] = pca.transform(X)
        Ys[i] = np.load(os.path.join(layer_folder_pre_processed, x_file_name.replace('_X_','_Y_')))[np.newaxis,:]
        
    np.save(os.path.join(layer_folder_ouput, "X"), np.concatenate(Xs, axis=0))
    np.save(os.path.join(layer_folder_ouput, "Y"), np.concatenate(Ys, axis=0))