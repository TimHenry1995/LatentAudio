import sys
sys.path.append(".")
from LatentAudio.configurations import loader as configuration_loader
import json, argparse, time
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
from LatentAudio.adapters import layer_wise as ylw
if __name__ == "__main__":
    

    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="explore_latent_yamnet",
        description='''This script loads the projections stored by the script apply_scaler_and_PCA_to_latent_yamnet at `pca_projected_folder` and for the given `factor_index`, explores samples 
                        (drawn uniformly at random without replacement and with random seed) of the latent yamnet representations by creating the following plots:
                        (1) A box-plot (with one box for each layer in `layer_indices`) for the accuracies of a cross-validated k-nearest neighbor (KNN) model that predicts the classes of 
                        the given factor based on the projections. The plot will be stored with the name 'Latent <factor name> KNN Classification` along with a text file called 'Latent <factor name> KNN Classification Statistics' storing the statistical comparison results between the first, most accurate and last layer.
                        (2) A scatterplot (with one subplot for the first, most-accurate (determined by KNN) and last layer) showing a projection from the previously projected latent 
                        representations down to 2D space using t-distributed stochastic neighborhood embeddings (t-SNE). Each dot will represent a sound and dots will be colored based on 
                        the classes of the factor. The plot will be stored with the name `Latent <factor name> t-SNE Projections`. Note, this step assumes that `layer_indices` contains at least 3 layer indices. 
                        All created plots are saved at `figure_folder`. If there is no pre-existing figure folder, it will be created. If the folder already contains files of same name,
                        they will will be renamed using the appendix ' (old) ' and a time stamp. 
                                                
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--pca_projected_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the projections are stored.", type=str)
    parser.add_argument("--layer_indices", help="A list containing the indices of the Yamnet layers for which the latent projections shall be explored.", type=str)
    parser.add_argument("--random_seeds", help="A list of integers indicating for each layer in the layer_indices list which seed shall be used to set the random module of numpy right before taking a sample of instances based on which the models will be calibrated. This argument helps wth reproducibility.", type=str)
    parser.add_argument("--sample_sizes", help="A list of integers (or None entries) indicating for each layer in the layer_indices list which sample size shall be used for KNN and t-SNE. For a given layer, the sample size is the overall sample size that will be split into `cross_validation_fols` many folds for KNN. That same overall sample will also be usd for t-SNE. An exception will be raised if the sample size is larger than the number of available instances.", type=str)
    parser.add_argument("--factor_name", help="The name of the factor of interest.", type=str)
    parser.add_argument("--factor_index", help="An int (in form of one json string) that gives the index of the factor of interest in the numeric label of a given instance. For example, if there are two factors (material and action) and the script is currently called on the action factor and the numeric label of a given instance codes the action factor at indxe 1 (zero-based), then factor_index should be set to 1 here.", type=str)
    parser.add_argument("--cross_validation_folds", help="The number of cross-validation folds (in form of a json string) used when fitting the KNN models.", type=str)
    parser.add_argument("--neighbor_count", help="The number of neighbors k (in form of a json string) used when fitting the KNN models.", type=str)
    parser.add_argument("--class_index_to_name", help="A dictionary (in form of a json string) that maps the class indices of the given factor to their corresponding class names. For instance, if the factor is the material of an instance and the its classes are wood, glass and metal (in that order). then the dictionary would map 0 to wood, 1 to glass and 2 to metal.")
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plots should be saved.")

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.pca_projected_folder != None and args.layer_indices != None and args.random_seeds != None and args.sample_sizes != None and args.factor_name != None and args.factor_index != None and args.cross_validation_folds != None and args.neighbor_count != None and args.class_index_to_name != None and args.figure_folder != None, "If no configuration file is provided, then all other arguments must be provided."
    
        pca_projected_folder = json.loads(args.pca_projected_folder)
        pca_projected_folder_path = os.path.join(*pca_projected_folder)
        layer_indices = json.loads(args.layer_indices)
        random_seeds = json.loads(args.random_seeds)
        sample_sizes = json.loads(args.sample_sizes)
        factor_name = args.factor_name
        factor_index = json.loads(args.factor_index)
        cross_validation_folds = json.loads(args.cross_validation_folds)
        neighbor_count = json.loads(args.neighbor_count)
        class_index_to_name = json.loads(args.class_index_to_name)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.pca_projected_folder == None and args.layer_indices == None and args.random_seeds == None and args.sample_sizes == None and args.factor_name == None and args.factor_index == None and args.cross_validation_folds == None and args.neighbor_count == None and args.class_index_to_name == None and args.figure_folder == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'explore_latent_yamnet' or configuration['script'] == 'explore_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'explore_latent_yamnet'."
        
        pca_projected_folder_path = os.path.join(*configuration['arguments']['pca_projected_folder'])
        layer_indices = configuration['arguments']['layer_indices']
        random_seeds = configuration['arguments']['random_seeds']
        sample_sizes = configuration['arguments']['sample_sizes']
        factor_name = configuration['arguments']['factor_name']
        factor_index = configuration['arguments']['factor_index']
        cross_validation_folds = configuration['arguments']['cross_validation_folds']
        neighbor_count = configuration['arguments']['neighbor_count']
        class_index_to_name = configuration['arguments']['class_index_to_name']
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])
        
    print("\n\n\tStarting script explore_latent_yamnet")
    print("\t\tThe script parsed the following arguments:")
    print("\t\tpca_projected_folder path: ", pca_projected_folder_path)
    print("\t\tlayer_indices: ", layer_indices)
    print("\t\trandom_seeds: ", random_seeds)
    print("\t\tsample_sizes: ", sample_sizes)
    print("\t\tfactor_name: ", factor_name)
    print("\t\tfactor_index: ", factor_index)
    print("\t\tcross_validation_folds: ", cross_validation_folds)
    print("\t\tneighbor_count: ", neighbor_count)
    print("\t\tclass_index_to_name: ", class_index_to_name)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\n\tStarting script now:\n")
    """
    pca_projected_folder_path = "E:\\LatentAudio\simple configuration\data\latent\pca projected"
    layer_indices = [0, 9, 13]
    random_seeds = [42, 42, 42]
    sample_sizes = [10000, 10000, 10000]
    factor_name = "Material"
    factor_index = 0
    cross_validation_folds = 10
    neighbor_count = 10
    class_index_to_name = {0: 'W', 1: 'M', 2: 'G',3: 'C'}
    figure_folder_path = "E:\\LatentAudio\simple configuration\\figures"
    """
    ### Start actual data processing

    # File management
    if not os.path.exists(figure_folder_path): os.makedirs(figure_folder_path)
    
    # For each layer
    projections = {}
    KNN_accuracies = {}
    Ys = {}
    print("\n\t\tLoading PCA projections of latent sound representations, fitting KNN and TSNE for layers ", end="")
    for l, layer_index in enumerate(layer_indices):
        # Log
        print(layer_index, end=" ")        
        
        # Load sample
        X = np.load(os.path.join(pca_projected_folder_path, f"Layer {layer_index}", "X.npy"))
        Y = np.load(os.path.join(pca_projected_folder_path, f"Layer {layer_index}", "Y.npy"))
        np.random.seed(random_seeds[l]) # Reproducibility
        sample_indices = np.random.randint(0,len(X),sample_sizes[l])
        X = X[sample_indices,:]
        Y = Y[sample_indices,factor_index]
        Ys[layer_index] = Y
        
        # Fit KNN
        KNN = KNeighborsClassifier(n_neighbors=neighbor_count)
        KNN_accuracies[layer_index] = cross_val_score(KNN, X, Y, cv=cross_validation_folds)

        # Fit t-SNE
        tsne = TSNE()
        projections[layer_index] = tsne.fit_transform(X)
    print(" completed.")

    # Determine most accurate layer
    most_accurate_layer_index = layer_indices[0]
    max_accuracy = 0
    for layer_index in layer_indices:
        max_accuracy_2 = np.median(KNN_accuracies[layer_index])
        if max_accuracy < max_accuracy_2:
            most_accurate_layer_index = layer_index
            max_accuracy = max_accuracy_2
    
    # Compute KNN Wilcoxon signed rank test to compare first, most accurate and last layer
    print(f"\n\t\tPerforming Wilcoxon signed rank tests to compare KNN predictions between layers.")
    first_index = np.min(layer_indices); last_index = np.max(layer_indices)
    statistical_results = {"test":f"Wilcoxon Signed Rank between {cross_validation_folds}-fold cross-validation scores of first, most accurate (by median accuracy) and last layer"}

    tmp = stats.wilcoxon(KNN_accuracies[first_index], KNN_accuracies[most_accurate_layer_index])
    statistical_results[f"Layer {first_index} versus {most_accurate_layer_index}"] = {"W": tmp.statistic, "p (Bonferroni corrected)": 3*tmp.pvalue}

    tmp = stats.wilcoxon( KNN_accuracies[most_accurate_layer_index], KNN_accuracies[last_index])
    statistical_results[f"Layer {most_accurate_layer_index} versus {last_index}"] = {"W": tmp.statistic, "p (Bonferroni corrected)": 3*tmp.pvalue}

    tmp = stats.wilcoxon(KNN_accuracies[first_index], KNN_accuracies[last_index])
    statistical_results[f"Layer {first_index} versus {last_index}"] = {"W": tmp.statistic, "p (Bonferroni corrected)": 3*tmp.pvalue}

    # Save results
    print_path = os.path.join(figure_folder_path, f"Latent {factor_name} KNN Classification Statistics.txt")
    if os.path.exists(print_path): 
        print(f"\t\tFound existing file at {print_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(print_path, print_path + ' (old) ' + (str)(time.time()))

    with open(print_path, 'w') as file:
        json.dump(statistical_results, file)

    # Plot KNN and TSNE
    
    # Plot KNN
    print(f"\t\tPlotting KNN accuracies and TSNE scatterplots.")
    plt.figure(figsize=(len(layer_indices),7)); plt.title(f"Latent {factor_name} KNN Classification")
    plt.boxplot([KNN_accuracies[layer_index] for layer_index in layer_indices], labels=[ylw.LayerWiseYamnet.layer_names[layer_index] for layer_index in layer_indices])
    
    # Add chance level line
    a = (int)(np.where(np.array(layer_indices)==first_index)[0][0])+1 # Used as x value for horizontal line
    b = (int)(np.where(np.array(layer_indices)==most_accurate_layer_index)[0][0])+1 # Used as x value for horizontal line
    c = (int)(np.where(np.array(layer_indices)==last_index)[0][0])+1 # Used as x value for horizontal line
    plt.ylim(1.0/len(class_index_to_name)-0.05,1.2); plt.ylabel('Accuracy'); plt.xlabel('Layer')
    plt.hlines([1.0/len(class_index_to_name)], xmin=a, xmax=c, color='red')
    plt.text(x=a, y=(1.0/len(class_index_to_name))+0.02,s='Chance Level')

    # Add lines and stars to KNN (with bonferroni correction)
    plt.hlines([1.05,1.1,1.15], xmin=[a, b, a], xmax=[b, c, c], color='black')
    if statistical_results[f"Layer {first_index} versus {most_accurate_layer_index}"]["p (Bonferroni corrected)"] < 0.05: plt.text((a+b)/2, 1.06, '*')
    if statistical_results[f"Layer {most_accurate_layer_index} versus {last_index}"]["p (Bonferroni corrected)"] < 0.05: plt.text((c+b)/2, 1.11, '*')
    if statistical_results[f"Layer {first_index} versus {last_index}"]["p (Bonferroni corrected)"] < 0.05: plt.text((a+c)/2, 1.16, '*')
    
    # Save figure
    knn_figure_path = os.path.join(figure_folder_path, f"Latent {factor_name} KNN Classification")
    if os.path.exists(knn_figure_path): 
        print(f"\t\tFound existing file at {knn_figure_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(knn_figure_path, knn_figure_path + ' (old) ' + (str)(time.time()))
    plt.tight_layout()
    plt.savefig(knn_figure_path)
    
    # Plot t-SNE
    plt.figure(figsize=(10,3))
    plt.suptitle(f"Latent {factor_name} t-SNE Projections")
    p=0 # subplot index
    for l, layer_index in {layer_indices.index(element): element for element in [first_index, most_accurate_layer_index, last_index]}.items():

        plt.subplot(1,3,p+1); p+= 1
        # Iterate classes of materials
        class_indices = set(list(np.reshape(Ys[layer_index], [-1])))
        for class_index in class_indices:
            is_instance_of_class = Ys[layer_index] == class_index
            plt.scatter(projections[layer_index][is_instance_of_class,0], projections[layer_index][is_instance_of_class,1], marker='.')
            plt.xlabel(f'Dimension 1\nLayer {ylw.LayerWiseYamnet.layer_names[layer_index]}')
        if l == 0:
            plt.legend([class_index_to_name[(str)((int)(class_index))] for class_index in class_indices])
            plt.ylabel('Dimension 2')
        plt.xticks([]); plt.yticks([])
    
    # Save figure
    tsne_figure_path = os.path.join(figure_folder_path, f"Latent {factor_name} t-SNE Projections")
    if os.path.exists(tsne_figure_path): 
        print(f"\t\tFound existing file at {tsne_figure_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(tsne_figure_path, tsne_figure_path + ' (old) ' + (str)(time.time()))
    plt.tight_layout()
    plt.savefig(tsne_figure_path)
    
    # Log
    print("\n\n\Completed script explore_latent_yamnet")