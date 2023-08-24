"""This module is used to explore the output of the pre_process script. It takes the latent representations of indicated layers and
uses principal component analaysos (PCA) to reduce the number of dimensions such that it is manageable for t-distributed stochastic
neighbor embeddings (t-SNE). It then applies t-SNE to visualize the classes of actions and the classes of materials.
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from sklearn.model_selection import cross_val_score
from scipy import stats
# Configuration

data_folder = os.path.join('data','pre-processed')
figure_output_folder = os.path.join('plots','data exploration')
sample_size = 2048 # The number of instances to be visualized
cross_validation_folds = 10
np.random.seed(42)
label_to_material = ['W','M','G','S','C','P']
label_to_action = ['T','R','D','W']

# Load layer indices
layer_indices = []
for folder_name in os.listdir(data_folder):
    if 'Layer' in folder_name: layer_indices.append((int)(folder_name.split(" ")[1])) # The word after the first space is assumed to be the layer number
layer_indices.sort()

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
    pca = PCA(n_components=0.8)
    pca.fit(X)
    explained_variances[layer_index] = pca.explained_variance_ratio_
    X = pca.transform(X)

    # Fit KNN
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN_accuracies[layer_index] = [cross_val_score(KNN, X, Y[:,0], cv=cross_validation_folds), cross_val_score(KNN, X, Y[:,1], cv=cross_validation_folds)] # Once for materials [0] and once for actions[1]

    # Fit t-SNE
    tsne = TSNE()
    projections[layer_index] = tsne.fit_transform(X)

# Plot KNN and TSE for both factors
for factor_index, factor_name, label_to_factor in zip([0,1], ['Material','Action'], [label_to_material, label_to_action]):

    # Plot KNN
    plt.figure(figsize=(10,7)); plt.title(f"K-Nearest Neighbor Classification on {factor_name}s")
    plt.boxplot([accuracies[factor_index] for accuracies in KNN_accuracies.values()], labels=KNN_accuracies.keys())
    plt.ylim(1.0/len(label_to_factor)-0.05,1.2); plt.ylabel('Accuracy'); plt.xlabel('Layer')
    plt.hlines([1.0/len(label_to_factor)], xmin=layer_indices[0]+1, xmax=layer_indices[-1]+1, color='red')
    plt.text(x=layer_indices[0]+1, y=(1.0/len(label_to_factor))+0.02,s='Chance Level')

    # Compute KNN significance between first, most accurate and last layer
    max_index = 0
    mean = 0
    for layer_index in KNN_accuracies.keys():
        mean_2 = np.mean(KNN_accuracies[layer_index][factor_index])
        if mean < mean_2:
          max_index = layer_index
          mean = mean_2

    first_index = np.min(layer_indices); last_index = np.max(layer_indices)
    _, p_first_to_max = stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[max_index][factor_index])
    _, p_max_to_last = stats.ttest_ind(KNN_accuracies[max_index][factor_index], KNN_accuracies[last_index][factor_index])
    _, p_first_to_last = stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[last_index][factor_index])

    # Add lines and stars to KNN (with bonferroni correction)
    plt.hlines([1,1.05,1.1], xmin=[first_index+1, max_index+1, first_index+1], xmax=[max_index+1, last_index+1, last_index+1], color='black')
    if p_first_to_max*3 < 0.05: plt.text((first_index+max_index+2)/2, 1.01, '*')
    if p_max_to_last*3 < 0.05: plt.text((last_index+max_index+2)/2, 0.99, '*')
    if p_first_to_max*3 < 0.05: plt.text((first_index+last_index+2)/2, 1.11, '*')

    plt.savefig(os.path.join(figure_output_folder, f"KNN {factor_name}"))

    # Plot t-SNE
    plt.figure(figsize=(10,3))
    plt.suptitle(f"T-Distributed Stochastic Neighbor Embeddings on {factor_name}s")
    p=0 # subplot index
    for l, layer_index in {layer_indices.index(element): element for element in [first_index, max_index, last_index]}.items():

        plt.subplot(1,3,p+1); p+= 1
        # Iterate classes of materials
        factor_labels = set(list(np.reshape(Ys[layer_index][:,factor_index], [-1])))
        for label in factor_labels:
            label_indices = Ys[layer_index][:, factor_index] == label
            plt.scatter(projections[layer_index][label_indices,0], projections[layer_index][label_indices,1], marker='.')
            plt.xlabel(f'Dimension 1\nLayer {layer_index}')
        if l == 0:
            plt.legend([label_to_factor[label] for label in factor_labels])
            plt.ylabel('Dimension 2')
        plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(figure_output_folder, f"TSNE {factor_name}"))

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
plt.savefig(os.path.join(figure_output_folder, f"PCA"))
plt.show()