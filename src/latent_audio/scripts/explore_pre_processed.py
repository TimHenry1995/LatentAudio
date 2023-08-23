"""This module is used to explore the output of the pre_process script. It takes the latent representations of indicated layers and 
uses principal component analaysos (PCA) to reduce the number of dimensions such that it is manageable for t-distributed stochastic 
neighbor embeddings (t-SNE). It then applies t-SNE to visualize the classes of actions and the classes of materials.
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
plt.rcParams["font.family"] = "Times New Roman"
from sklearn.model_selection import cross_val_score

# Configuration
factor_name = 'Material' # or alternatively 'Action'
data_folder = os.path.join('data','pre-processed')
sample_size = 128 # The number of instances to be visualized
np.random.seed(42)
label_to_material = ['W','M','G','S','C','P']
label_to_action = ['T','R','D','W']
if factor_name == 'Material': 
    label_to_factor = label_to_material
    factor_index = 0
else: 
    label_to_factor = label_to_action
    factor_index = 1

# Load layer indices
layer_indices = []
for folder_name in os.listdir(data_folder):
    if 'Layer' in folder_name: layer_indices.append((int)(folder_name.split(" ")[1])) # The word after the first space is assumed to be the layer number
layer_indices.sort()
layer_indices = [0,6,13]
# For each layer
explained_variances = {}
projections = {}
KNN_accuracies = {}
Ys = {}

for layer_index in layer_indices:
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
    Ys[layer_index] = Y

    # Fit PCA 
    pca = PCA(n_components=50)
    pca.fit(X)
    explained_variances[layer_index] = pca.explained_variance_ratio_
    X = pca.transform(X)
    
    # Fit KNN
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN_accuracies[layer_index] = cross_val_score(KNN, X, Y[:,factor_index], cv=5)
    
    # Fit t-SNE
    tsne = TSNE()
    projections[layer_index] = tsne.fit_transform(X)
    
# Plot t-SNE
plt.figure(figsize=(10,3))
plt.suptitle(f"T-Distributed Stochastic Neighbor Embeddings on {factor_name}s")
p=0 # subplot index
for l, layer_index in {layer_indices.index(element): element for element in [0,6,13]}.items():
    
    plt.subplot(1,3,p+1); p+= 1
    # Iterate classes of materials
    factor_labels = set(list(np.reshape(Ys[layer_index][:,1], [-1])))
    for label in factor_labels:
        label_indices = Ys[layer_index][:, factor_index] == label
        plt.scatter(projections[layer_index][label_indices,0], projections[layer_index][label_indices,1], marker='.')
        plt.xlabel(f'Dimension 1\nLayer {layer_index}')
    if l == 0: 
        plt.legend([label_to_factor[label] for label in factor_labels])
        plt.ylabel('Dimension 2')
    plt.xticks([]); plt.yticks([])

# Plot KNN
plt.figure(figsize=(10,5)); plt.title(f"K-Nearest Neighbor Classification on {factor_name}s")
plt.boxplot(KNN_accuracies.values(), labels=KNN_accuracies.keys())
plt.ylim(0,1); plt.ylabel('Accuray'); plt.xlabel('Layer')

# Plot PCA
plt.figure(figsize=(10,5)); plt.title("Principal Component Analysis")
for layer_index in layer_indices:
    R = 0
    plt.gca().set_prop_cycle(None)
    for i, r in enumerate(explained_variances[layer_index]):
        plt.bar([str(layer_index)],[r], bottom=R, color='white', edgecolor='black')
        R += r
    
    plt.ylim(0,1)

plt.ylabel('Explained Variance'); plt.xlabel('Layer')
plt.show()