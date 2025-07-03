import sys
sys.path.append(".")
from LatentAudio import utilities as utl
from LatentAudio.configurations import loader as configuration_loader
import os, pickle as pkl, numpy as np, shutil, argparse, json, time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import LatentAudio.adapters.layer_wise as ylw

if __name__ == "__main__":

    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
        description='''This script assumes that the `audio_to_latent_yamnet` and `create_scaler_and_PCA_model_for_latent_yamnet` scripts were executed beforehand. 
                        Then, if the `projection_folder` already exists, it will be renamed with appendix '(old) ' and a time-stamp. The script then iterates
                        all layers in `layer_indices` and loads the corresponding latent representations of original dimensionality from the `latent_representations_folder` 
                        as well as the standard scalers and PCA model from `PCA_and_standard_scaler_folder` and projects the flattened data down to the desired dimensionality given by `target_dimensionalities`.
                        Then, the script creates a new folder named by that layer inside the `projection_folder` and saves the projections as one large file called 'X.npy' of shape [instance count, target_dimensionality]
                        Next to that, the script also maps the file names from the latent representations to factor-wise numeric labels using `file_name_prefix_to_factor_wise_label`
                        and stores them in one large file called `Y.npy` of shape [instance count, factor count].
                        Furthermore, the script creates a stacked bar-chart to illustrate the proportion of explained variance per projected dimension (one bar stacks the variance for all included components) for each layer (one bar per layer).
                        The figure will be saved at `figure_folder` using the title 'PCA - Explained Variances'. If that file already exists, it will be renamed using the 
                        appendix ' (old) ' and a time-stamp before the new file is saved.
                        
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--latent_representations_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations are stored. The files are expected to be stored exactly as done by the audio_to_latent_yamnet script.", type=str)
    parser.add_argument("--PCA_and_standard_scaler_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the models are stored.", type=str)
    parser.add_argument("--layer_indices", help="A list containing the indices of the Yamnet layers for which the projected data sets shall be computed.", type=str)
    parser.add_argument("--target_dimensionalities", help="A list of integers (or None entries) indicating for each layer in the layer_indices list to which dimensionality the projection shall be made. This needs to be at most as large as what the corresponding pre-trained PCA model supports.", type=str)
    parser.add_argument("--projection_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the outputs shall be stored.", type=str)
    parser.add_argument("--file_name_prefix_to_factor_wise_label", help="A dictionary (as json string) that maps from a prefix of the file name of a latent representation to its factorwise numeric labels. For example, if a latent representation of a single sound is called CD_X_1.npy, then 'CD' would be be mapped to [1,3], if C is the abbreviation for the first factor's class whose index is 1 and D the second factor's class whose index is 3. Note that the factor-wise numeric labels do not include the residual factor but only the actual factors that would be of interest to a flow model.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot of explained variances should be saved.", type=str)

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.latent_representations_folder != None and args.PCA_and_standard_scaler_folder != None and args.layer_indices != None and args.target_dimensionalities != None and args.projection_folder != None and args.file_name_prefix_to_factor_wise_label != None and args.figure_folder != None, "If no configuration file is provided, then all other arguments must be provided."
    
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        PCA_and_standard_scaler_folder = json.loads(args.PCA_and_standard_scaler_folder)
        PCA_and_standard_scaler_folder_path = os.path.join(*PCA_and_standard_scaler_folder)
        layer_indices = json.loads(args.layer_indices)
        target_dimensionalities = json.loads(args.target_dimensionalities)
        projection_folder = json.loads(args.projection_folder)
        projection_folder_path = os.path.join(*projection_folder)
        file_name_prefix_to_factor_wise_label = json.loads(args.file_name_prefix_to_factor_wise_label)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert  args.latent_representations_folder == None and args.PCA_and_standard_scaler_folder == None and args.layer_indices == None and args.target_dimensionalities == None and args.projection_folder == None and args.file_name_prefix_to_factor_wise_label == None and args.figure_folder == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'apply_scalers_and_PCA_to_latent_yamnet' or configuration['script'] == 'apply_scalers_and_PCA_to_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'apply_scalers_and_PCA_to_latent_yamnet'."
        
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        PCA_and_standard_scaler_folder_path = os.path.join(*configuration['arguments']['PCA_and_standard_scaler_folder'])
        layer_indices = configuration['arguments']['layer_indices']
        target_dimensionalities = configuration['arguments']['target_dimensionalities']
        projection_folder_path = os.path.join(*configuration['arguments']['projection_folder'])
        file_name_prefix_to_factor_wise_label = configuration['arguments']['file_name_prefix_to_factor_wise_label']
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])
        
    print("\n\n\tStarting script apply_scalers_and_PCA_to_latent_yamnet")
    print("\t\tThe script parsed the following arguments:")
    print("\t\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\t\tPCA_and_standard_scaler_folder path: ", PCA_and_standard_scaler_folder_path)
    print("\t\tlayer_indices: ", layer_indices)
    print("\t\ttarget_dimensionalities: ", target_dimensionalities)
    print("\t\tprojection_folder path: ", projection_folder_path)
    print("\t\tfile_name_prefix_to_factor_wise_label: ", file_name_prefix_to_factor_wise_label)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\n\tStarting script now:\n")

    ### Start actual data processing
    
    # Rename prexisting output folder
    if os.path.exists(projection_folder_path):
        print(f"\t\tFound existing folder at {projection_folder_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(projection_folder_path, projection_folder_path + ' (old) ' + str(time.time()))
        
    # Iterate layers
    explained_variances = [None] * len(layer_indices)
    for l, layer_index in enumerate(layer_indices):
        print(f"\n\t\tLayer {layer_index}.")

        # Manage path for layer
        latent_representations_folder_layer_path = os.path.join(latent_representations_folder_path, f'Layer {layer_index}')
        PCA_and_standard_scaler_folder_layer_path = os.path.join(PCA_and_standard_scaler_folder_path, f"Layer {layer_index}")
        
        # Loads models
        with open(os.path.join(PCA_and_standard_scaler_folder_layer_path, "Pre PCA Standard Scaler.pkl"),'rb') as fh:
            pre_scaler = pkl.load(fh)

        with open(os.path.join(PCA_and_standard_scaler_folder_layer_path, "PCA.pkl"),'rb') as fh:
            pca = pkl.load(fh)
            # Extract explained proportion of variance for plotting
            target_dimensionality = target_dimensionalities[l]
            explained_variances[l] = pca.explained_variance_ratio_[:target_dimensionality]
        
        with open(os.path.join(PCA_and_standard_scaler_folder_layer_path, "Post PCA Standard Scaler.pkl"), 'rb') as fh:
            post_scaler = pkl.load(fh)

        # Remove unwanted file names
        x_file_names = utl.find_matching_strings(strings=os.listdir(latent_representations_folder_layer_path), token='_X_')
        for x_file_name in reversed(x_file_names):
            found = False
            for label in file_name_prefix_to_factor_wise_label.keys(): 
                if label in x_file_name: found = True
            if not found: x_file_names.remove(x_file_name)

        # Iterate original latent representations
        x_file_names.sort() # Needed to ensure that the file names are consistently ordered across runs
        Xs = np.zeros([len(x_file_names), target_dimensionality]); 
        Ys = np.zeros([len(x_file_names), len(list(file_name_prefix_to_factor_wise_label.values())[0])])
        
        for i, x_file_name in enumerate(x_file_names):
            
            # Transform them
            X = np.load(os.path.join(latent_representations_folder_layer_path, x_file_name))[np.newaxis,:]
            Xs[i,:] = post_scaler.transform(pca.transform(pre_scaler.transform(X)))[:,:target_dimensionality] 
            
            # Determine factorwise labels
            file_name_prefix = x_file_name.split(sep='_X_')[0]
            factorwise_label = file_name_prefix_to_factor_wise_label[file_name_prefix]
            Ys[i,:] = np.array(factorwise_label)
        
            print(f"\r\t{np.round(100*(i+1)/len(x_file_names))} % completed", end='')
        
        # Save files
        projection_folder_layer_path = os.path.join(projection_folder_path, f'Layer {layer_index}')
        os.makedirs(projection_folder_layer_path)
        np.save(os.path.join(projection_folder_layer_path, "X"), Xs)
        np.save(os.path.join(projection_folder_layer_path, "Y"), Ys)

    # Prepare plot for proportion of variance in the original data that is explained by the variance that is in the projection
    print(f"\n\t\tCreating figure for proportion of explained variances now.")
    plt.figure(figsize=(len(layer_indices),5)); plt.title("PCA - Explained Variances")
    
    # Iterate the layers
    for l, layer_index in enumerate(layer_indices):
        
        # Plot the proportion of variance
        plt.gca().set_prop_cycle(None)
        R = 0
        for i, r in enumerate(explained_variances[l]):
            plt.bar([ylw.LayerWiseYamnet.layer_names[layer_index]],[r], bottom=R, color='white', edgecolor='black')
            R += r

        plt.ylim(0,1)
    plt.ylabel('Explained Variance'); plt.xlabel('Layer')
    
    # Save figure
    if not os.path.exists(figure_folder_path): os.makedirs(figure_folder_path)
    figure_path = os.path.join(figure_folder_path, "PCA - Explained Variances")
    if os.path.exists(figure_path): 
        print(f"\t\tFound existing figure at {figure_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(figure_path, figure_path + ' (old) ' + (str)(time.time()))
    plt.savefig(figure_path)
    
    # Log
    print("\n\n\Completed script apply_scalers_and_PCA_to_latent_yamnet")
            