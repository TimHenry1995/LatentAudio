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

def run(sample_size = 2048, latent_data_folder:str=None, figure_output_folder:str=None):
    """
    This function explores the layerwise yamnets representations with several plots.

    Requirements:
    - the audio_to_latent_yamnet script needs to be executed apriori

    Steps:
    - For all layers it loads the latent representations
    - creates a custom small standard scaler and pca model (thus does NOT use the complete one from the create_scalers_and_model_for_latent_yamnet script because running that one on all layers would tak etoo long)
    - projects them down to a size manageable for KNN and TSNE 
    - For each layer it fits a cross validated KNN to predict actions and materials
    - For the first, maximally accurate and last layer it plots the TSNE 2D Scatter plots
    - Creates a bar plot to visualize the PCA explained proportion of variance per Yament layer
    - Saves the plots

    Side effects:
    - Any eqaully named previosuly created plots will be overridden.

    :type sample_size: int, optional
    :param latent_data_folder: The folder from which the latent data shall be drawn. If None, then the default folder inside the internal file system will be used (default is None).
    :type latent_data_folder: str, optional
    :param figure_output_folder: The folder where the output figures shall be saved. If None, then the default folder inside the internal file system will be used (default is None).
    :type figure_output_folder: str, optional
    """


    # Configuration
    np.random.seed(42)
    latent_data_folder = os.path.join('data','latent yamnet', '64 dimensions')
    figure_output_folder = os.path.join('plots','explore latent yamnet','64 dimensions')
    cross_validation_folds = 10
    label_to_material = ['W','M','G','S','C','P']
    label_to_action = ['T','R','D','W']

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
    print(" Completed.")


    # Plot KNN and TSE for both factors
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
        plt.hlines([1.0/len(label_to_factor)], xmin=layer_indices[0]+1, xmax=layer_indices[-1]+1, color='red')
        plt.text(x=layer_indices[0]+1, y=(1.0/len(label_to_factor))+0.02,s='Chance Level')

        # Add lines and stars to KNN (with bonferroni correction)
        plt.hlines([1,1.05,1.1], xmin=[first_index+1, max_index+1, first_index+1], xmax=[max_index+1, last_index+1, last_index+1], color='black')
        if p_first_to_max*3 < 0.05: plt.text((first_index+max_index+2)/2, 1.01, '*')
        if p_max_to_last*3 < 0.05: plt.text((last_index+max_index+2)/2, 0.99, '*')
        if p_first_to_last*3 < 0.05: plt.text((first_index+last_index+2)/2, 1.11, '*')
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
    run()
    m=2