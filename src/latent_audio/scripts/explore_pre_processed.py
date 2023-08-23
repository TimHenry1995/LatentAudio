"""This module is used to explore the output of the pre_process script. It takes the latent representations of indicated layers and 
uses principal component analaysos (PCA) to reduce the number of dimensions such that it is manageable for t-distributed stochastic 
neighbor embeddings (t-SNE). It then applies t-SNE to visualize the classes of actions and the classes of materials.
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Configuration
data_folder = os.path.join('data','pre-processed')
sample_size = 500 # The number of instances to be visualized
np.random.seed(42)
label_to_material = ['W','M','G','S','C','P']
label_to_action = ['T','R','D','W']

# Load layer indices
layer_indices = []
for folder_name in os.listdir(data_folder):
    if 'Layer' in folder_name: layer_indices.append((int)(folder_name.split(" ")[1])) # The word after the first space is assumed to be the layer number
layer_indices.sort()

# For each layer
explained_variances = [None] * len(layer_indices)
projections = [None] * len(layer_indices)
Ys = [None] * len(layer_indices)
Y_hats = [None] * len(layer_indices)
for l, layer_index in enumerate(layer_indices):
    # Convenience variables
    layer_name = f"Layer {layer_index}"

    # Load sample
    x_file_name_count = np.sum([1 if "_X_" in file_name else 0 for file_name in os.listdir(os.path.join(data_folder, layer_name))])
    x_file_names = [None] * x_file_name_count
    i = 0
    for file_name in os.listdir(os.path.join(data_folder, layer_name)):
        if '_X_' in file_name: 
            x_file_names[i] = file_name; i+= 1

    X = [None] * sample_size; Y = [None] * sample_size
    for i, j in enumerate(np.random.randint(low=0, high=x_file_name_count, size=sample_size)):
        x_path = os.path.join(data_folder, layer_name, str(x_file_names[j]))
        X[i] = np.load(x_path)[np.newaxis,:]; Y[i] = np.load(x_path.replace('_X_','_Y_'))[np.newaxis,:]
    X = np.concatenate(X, axis=0); Y = np.concatenate(Y, axis=0)
    Ys[l] = Y

    # Fit PCA 
    pca = PCA(n_components=50)
    pca.fit(X)
    explained_variances[l] = pca.explained_variance_ratio_
    X = pca.transform(X)
    
    # Fit KNN
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X[:(int)(0.8*X.shape[0]),:], Y[:(int)(0.8*X.shape[0]),:])
    Y_hats[l] = KNN.predict(X[(int)(0.8*X.shape[0]):,:])

    # Fit t-SNE
    tsne = TSNE()
    projections[l] = tsne.fit_transform(X)
    
# Plot t-SNE
plt.figure(figsize=(10,8)); plt.suptitle("Standard Material Separation")
p=0 # subplot index
for l, layer_index in {layer_indices.index(element): element for element in [0,6,13]}.items():
    
    plt.subplot(3,3,p+1); p+= 1
    # Iterate classes of materials
    material_labels = set(list(np.reshape(Ys[l][:,1], [-1])))
    for label in material_labels:
        label_indices = Ys[l][:,1] == label
        plt.scatter(projections[l][label_indices,0], projections[l][label_indices,1])
    if l == 0: 
        plt.legend([label_to_action[label] for label in material_labels])
        plt.ylabel('TSNE\n\n\n')
    plt.xticks([]); plt.yticks([])

# Evaluate KNN
material_accuracies = [None] * len(layer_indices)
for l, layer_index in enumerate(layer_indices):
    material_accuracies[l] = accuracy_score(Ys[l][(int)(0.8*X.shape[0]):,1], Y_hats[l][:,1]) # Materials are at index 0

# Plot KNN
plt.subplot(3,1,2)
plt.bar([str(i) for i in layer_indices], material_accuracies)
plt.ylim(0,1); plt.ylabel('KNN\nAccuray')

# Plot PCA
plt.subplot(3,1,3)
for l, layer_index in enumerate(layer_indices):
    R = 0
    plt.gca().set_prop_cycle(None)
    for i, r in enumerate(explained_variances[l]):
        plt.bar([str(layer_index)],[r], bottom=R)
        R += r
    
    plt.ylim(0,1)

plt.xlabel('Layer'); plt.ylabel('PCA\nExplained Variance')
plt.show()