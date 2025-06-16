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
from LatentAudio import utilities as utl
from typing import List

def run(latent_data_folder:str, 
        figure_output_folder:str, 
        label_to_material: List[str], 
        label_to_action: List[str],
        random_seed: int,
        sample_size: int = 2048,
        cross_validation_folds:int = 10
        ):
    """
    This function loads the layerwise latent yamnet projections, fits a cross-validated k-nearest neighbor model to each to predict materials and actions and creates a 2D projection for the first, most accurate and last layers using t-distributed stochastic neighborhood embeddings (t-SNE). 
    All created plots are saved at `figure_output_folder` where they replace any existing plots of same name (if existent). Assumes that the latent_yamnet_to_calibration_data_set.run() function is executed beforehand. 

    :param latent_data_folder: The folder from which the latent data shall be drawn. If None, then the default folder inside the internal file system will be used (default is None).
    :type latent_data_folder: str
    :param figure_output_folder: The folder where the output figures shall be saved. If None, then the default folder inside the internal file system will be used (default is None).
    :type figure_output_folder: str
    :param label_to_material: The list of material labels.
    :type label_to_material: List[str]
    :param label_to_action: The list of action labels.
    :type label_to_action: List[str]
    :param sample_size: The number of instance that shall be loaded for KNN and t-SNE.
    :type sample_size: int, optional
    :param cross_validation_folds: The number of cross validation folds that shall be used.
    :type cross_validation_folds: int, optional
    :param random_seed: The seed used to set the random module of python before selecting random instances to be fed through the models.
    :type random_seed: int
    """
    np.random.seed(random_seed)

    # Load layer indices
    layer_indices = []
    for folder_name in os.listdir(latent_data_folder):
        if 'Layer' in folder_name: layer_indices.append((int)(folder_name.split(" ")[1])) # The word after the first space is assumed to be the layer number
    layer_indices.sort()

    # For each layer
    projections = {}
    KNN_accuracies = {}
    Ys = {}
    print("Loading PCA projections of latent sound representations, fitting KNN and TSNE for layers ", end="")
    for layer_index in layer_indices:
        # Load sample
        X = np.load(os.path.join(latent_data_folder, f"Layer {layer_index}", "X.npy"))
        Y = np.load(os.path.join(latent_data_folder, f"Layer {layer_index}", "Y.npy"))
        sample_indices = np.random.randint(0,len(X),sample_size)
        X = X[sample_indices,:]; Y = Y[sample_indices,:]
        Ys[layer_index] = Y
        print(layer_index, end=" ")
        
        # Fit KNN
        KNN = KNeighborsClassifier(n_neighbors=3)
        KNN_accuracies[layer_index] = [cross_val_score(KNN, X, Y[:,0], cv=cross_validation_folds), cross_val_score(KNN, X, Y[:,1], cv=cross_validation_folds)] # Once for materials [0] and once for actions[1]

        # Fit t-SNE
        tsne = TSNE()
        projections[layer_index] = tsne.fit_transform(X)
    print(" completed.")


    # Plot KNN and TSNE for both factors
    if not os.path.exists(figure_output_folder): os.makedirs(figure_output_folder)
    for factor_index, factor_name, label_to_factor in zip([0,1], ['Material','Action'], [label_to_material, label_to_action]):
        print(f"Processing factor {factor_name}")
        print(f"\tPerforming significance tests for KNN predictions.")
        
        # Compute KNN significance between first, most accurate and last layer
        max_index = 0
        mean = 0
        for layer_index in KNN_accuracies.keys():
            mean_2 = np.mean(KNN_accuracies[layer_index][factor_index])
            if mean < mean_2:
                max_index = layer_index
                mean = mean_2
        print("\tThe layer for which KNN recognizes classes of this factor most accurately has index ", max_index)
        first_index = np.min(layer_indices); last_index = np.max(layer_indices)
        t_first_to_max, p_first_to_max = stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[max_index][factor_index])
        t_max_to_last, p_max_to_last = stats.ttest_ind(KNN_accuracies[max_index][factor_index], KNN_accuracies[last_index][factor_index])
        t_first_to_last, p_first_to_last = stats.ttest_ind(KNN_accuracies[first_index][factor_index], KNN_accuracies[last_index][factor_index])

        # Print results
        print(f"\tSignificance test results:")
        print("\t\tComparing accuracy for first layer and most accurate layer:")
        print(f"\t\t\tt-statistic: {t_first_to_max}, p-value: {p_first_to_max*3} (Bonferroni corrected)")
        print("\t\tComparing accuracy for most accurate layer and last layer:")
        print(f"\t\t\tt-statistic: {t_max_to_last}, p-value: {p_max_to_last*3} (Bonferroni corrected)")
        print("\t\tComparing accuracy for first and last layer:")
        print(f"\t\t\tt-statistic: {t_first_to_last}, p-value: {p_first_to_last*3} (Bonferroni corrected)")
        
        # Plot KNN
        print(f"\tPlotting KNN accuracies and TSNE scatterplots.")
        plt.figure(figsize=(10,7)); plt.title(f"K-Nearest Neighbor Classification on {factor_name}s")
        plt.boxplot([accuracies[factor_index] for accuracies in KNN_accuracies.values()], labels=KNN_accuracies.keys())
        plt.ylim(1.0/len(label_to_factor)-0.05,1.2); plt.ylabel('Accuracy'); plt.xlabel('Layer')
        a = (int)(np.where(np.array(layer_indices)==first_index)[0])+1 # Used as x value for horizontal line
        b = (int)(np.where(np.array(layer_indices)==max_index)[0])+1 # Used as x value for horizontal line
        c = (int)(np.where(np.array(layer_indices)==last_index)[0])+1 # Used as x value for horizontal line
        plt.hlines([1.0/len(label_to_factor)], xmin=a, xmax=c, color='red')
        plt.text(x=a, y=(1.0/len(label_to_factor))+0.02,s='Chance Level')

        # Add lines and stars to KNN (with bonferroni correction)
        plt.hlines([1,1.05,1.1], xmin=[a, b, a], xmax=[b, c, c], color='black')
        if p_first_to_max*3 < 0.05: plt.text((a+b+2)/2, 1.01, '*')
        if p_max_to_last*3 < 0.05: plt.text((c+b+2)/2, 0.99, '*')
        if p_first_to_last*3 < 0.05: plt.text((a+c+2)/2, 1.11, '*')
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

        print(f"\t\tThe plots can be found at {figure_output_folder}")

if __name__ == "__main__":
    
    # Load Configuration
    import json, os
    with open(os.path.join('LatentAudio','configuration.json'),'r') as f:
        configuration = json.load(f)

    run(latent_data_folder=os.path.join(configuration['latent_yamnet_folder'],'projected', f'{configuration['PCA_target_dimensionality']} dimensions'),
        figure_output_folder=os.path.join(configuration['plots_folder'],'explore latent yamnet', f'{configuration['PCA_target_dimensionality']} dimensions'),
        cross_validation_folds = configuration['knn_cross_validation_folds'],
        label_to_material = {value : key for key, value in configuration['material_to_index']},
        label_to_action = {value : key for key, value in configuration['action_to_index']},
        sample_size=configuration['knn_tSNE_sample_size'],
        random_seed=configuration['random_seed']
        )