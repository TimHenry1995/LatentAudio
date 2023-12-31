"""This script explores the layerwise yamnets representations with some plots.

Requirements:
- the audio_to_latent_yamnet script needs to be executed apriori

Steps:
- For all layers it laods the latent representations
- creates a custom small standard scaler and pca model (thus does NOT use the complete one from the create_scalers_and_model_for_latent_yamnet script because running that one on all layers would tak etoo long)
- projects them down to a size manageable for KNN and TSNE 
- For each layer it fits a cross validated KNN to predict actions and materials
- For the first, maximally accurate and last layer it plots the TSNE 2D Scatter plots
- Saves the plots

Side effects:
- Any eqaully named previosuly created plots will be overridden.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from sklearn.model_selection import cross_val_score
from scipy import stats
from latent_audio import utilities as utl

# Configuration
pca_dimensionality = 64
latent_data_folder = os.path.join('data','latent yamnet', 'original')
figure_output_folder = os.path.join('plots','explore latent yamnet',f'{pca_dimensionality} dimensions')
sample_size = 2048 # The number of instances to be visualized
cross_validation_folds = 10
np.random.seed(42)
label_to_material = ['W','M','G','S','C','P']
label_to_action = ['T','R','D','W']

# Load layer indices
layer_indices = []
for folder_name in os.listdir(latent_data_folder):
    if 'Layer' in folder_name: layer_indices.append((int)(folder_name.split(" ")[1])) # The word after the first space is assumed to be the layer number
layer_indices.sort()

# For each layer
explained_variances = {}
projections = {}
KNN_accuracies = {}
dimensionalities = {}
Ys = {}

for layer_index in layer_indices:
    print(f"Layer {layer_index}")

    # Load sample
    X, Y = utl.load_latent_sample(data_folder=os.path.join(latent_data_folder, f"Layer {layer_index}"), sample_size=sample_size)
    Ys[layer_index] = Y
    print(f"\tLoaded sample: X shape == {X.shape}, Y shape == {Y.shape}")
    dimensionalities[layer_index] = X.shape[1] # X is assumed to be of shape [instances, dimensions]

    # Fit Scalers and PCA
    pre_scaler = StandardScaler()
    pca = PCA(n_components=pca_dimensionality)
    post_scaler = StandardScaler()
    X = post_scaler.fit_transform(pca.fit_transform(pre_scaler.fit_transform(X)))
    explained_variances[layer_index] = pca.explained_variance_ratio_
    print("\tApplied pre PCA Standard Scaler, PCA and post PCA Standard Scaler")

    # Fit KNN
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN_accuracies[layer_index] = [cross_val_score(KNN, X, Y[:,0], cv=cross_validation_folds), cross_val_score(KNN, X, Y[:,1], cv=cross_validation_folds)] # Once for materials [0] and once for actions[1]

    # Fit t-SNE
    tsne = TSNE()
    projections[layer_index] = tsne.fit_transform(X)

# Plot KNN and TSE for both factors
if not os.path.exists(figure_output_folder): os.makedirs(figure_output_folder)
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

    print(factor_name)
    print("First to Max Test:")
    print(stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[max_index][factor_index]))
    print("Max to Last Test:")
    print(stats.ttest_ind(KNN_accuracies[max_index][factor_index], KNN_accuracies[last_index][factor_index]))
    print("First to Last Test:")
    print(stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[last_index][factor_index]))
    """
    Material
    First to Max Test:
    TtestResult(statistic=-24.49502270898721, pvalue=2.8350567781716908e-15, df=18.0)
    Max to Last Test:
    TtestResult(statistic=15.667175554321526, pvalue=6.209791280475332e-12, df=18.0)
    First to Last Test:
    TtestResult(statistic=-4.208217189965692, pvalue=0.0005284838784518604, df=18.0)
    Action
    First to Max Test:
    TtestResult(statistic=-15.569876329499271, pvalue=6.8959880525901215e-12, df=18.0)
    Max to Last Test:
    TtestResult(statistic=11.096090030512535, pvalue=1.7613527629498179e-09, df=18.0)
    First to Last Test:
    TtestResult(statistic=-7.999791634899012, pvalue=2.4516297697215936e-07, df=18.0)
    """
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
plt.figure(figsize=(10,5)); plt.title(f"Principal Component Analysis")
for layer_index in layer_indices:
    R = 0
    plt.gca().set_prop_cycle(None)
    for i, r in enumerate(explained_variances[layer_index]):
        plt.bar([str(layer_index)],[r], bottom=R, color='white', edgecolor='black')
        R += r

    plt.ylim(0,1)
plt.ylabel('Explained Variance'); plt.xlabel('Layer')
plt.savefig(os.path.join(figure_output_folder, f"Principal Component Analysis"))
plt.show()

# Plot dimensionalities
plt.figure(figsize=(10,5)); plt.title("Original Dimensionalities")
plt.bar(dimensionalities.keys(), dimensionalities.values(), color='white', edgecolor='black')
plt.ylabel("Dimensionality"); plt.xlabel('Layer')
plt.savefig(os.path.join(figure_output_folder, "Original Dimensionalities"))
plt.show()

