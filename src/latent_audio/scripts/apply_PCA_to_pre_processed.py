"""This script takes the data as output by the pre_process script and applies a principal component analysis to each layer 
to reduce the dimensionality. This is done to allow for equal model sizes within the disentangle script."""

import os
from latent_audio import utilities as utl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl

# Configuration
pre_processed_folder = "/Volumes/Untitled 2/pre-processed"#os.path.join('data','pre-processed','All PCA dimensions')
pca_dimensionality = 64
output_folder = os.path.join('data', 'pre-processed',f'{pca_dimensionality} PCA dimensions all in 1 file')
model_folder = os.path.join('models')
sample_size = 2048

for l in [8]:#range(14):
    np.random.seed(42)
    layer_folder_pre_processed = os.path.join(pre_processed_folder, f'Layer {l}')
    layer_folder_ouput = os.path.join(output_folder, f'Layer {l}')
    layer_folder_model = os.path.join(model_folder, f"Layer {l}")
    if not os.path.exists(layer_folder_ouput): os.makedirs(layer_folder_ouput)
    if not os.path.exists(layer_folder_model): os.makedirs(layer_folder_model)
    
    # Load sample
    scaler = StandardScaler()
    X_sample, Y_sample = utl.load_latent_sample(data_folder=layer_folder_pre_processed, sample_size=sample_size)
    X_sample = scaler.fit_transform(X_sample)

    # Fit PCA
    pca = PCA(n_components=pca_dimensionality)
    pca.fit(X_sample)
    '''
    # Transform each X
    x_file_names = utl.find_matching_strings(strings=os.listdir(layer_folder_pre_processed), token='_X_')
    Xs = [None] * len(x_file_names); Ys = [None] * len(x_file_names)
    for i, x_file_name in enumerate(x_file_names):
        X = np.load(os.path.join(layer_folder_pre_processed, x_file_name))[np.newaxis,:]
        Xs[i] = pca.transform(scaler.transform(X))
        Ys[i] = np.load(os.path.join(layer_folder_pre_processed, x_file_name.replace('_X_','_Y_')))[np.newaxis,:]
        
    # Save
    np.save(os.path.join(layer_folder_ouput, "X"), np.concatenate(Xs, axis=0))
    np.save(os.path.join(layer_folder_ouput, "Y"), np.concatenate(Ys, axis=0))'''
    with open(os.path.join(layer_folder_model, "Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(scaler, file_handle)
        
    with open(os.path.join(layer_folder_model, f"PCA {pca_dimensionality}.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)