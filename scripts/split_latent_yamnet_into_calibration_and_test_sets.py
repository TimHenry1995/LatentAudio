import sys
sys.path.append(".")
from LatentAudio import utilities as utl
from LatentAudio.configurations import loader as configuration_loader
import os, pickle as pkl, numpy as np, shutil, argparse, json, time
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
        description='''This script takes. for each layer named in `layer_indices`, the latent yamnet data as saved by `audio_to_latent_yamnet` to `latent_representations_folder` and splits hem into two disjoint random subsets with relative proportions
                        1-`test_proportion` and `test_proportion`. It then creates two new folders (one at `calibration_folder` and one at `test_folder`) to which it moves
                        these two respective subsets. As a result, the original `latent_representations_folder` will be empty. The script throws an error in case either of the
                        two output folders already exist and aborts before saving any files to them. Note that the same instances will be moved for each layer and hence it is assumed that 
                        the same instances exist for every named layer in the source folder.

                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--latent_representations_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations are stored. The files are expected to be stored exactly as done by the audio_to_latent_yamnet script.", type=str)
    parser.add_argument("--calibration_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the calibration set shall be stored.", type=str)
    parser.add_argument("--test_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the test set shall be stored.", type=str)
    parser.add_argument("--layer_indices", help="A list containing the indices of the Yamnet layers for which the splitting shall be performed.", type=str)
    parser.add_argument("--random_seed", help="An int (in form of a json string) that will be used to set the random module of Python before sampling the test set.", type=str)
    parser.add_argument("--test_proportion", help="A float (in form of a json string) that indicates the proportion of instances that should be taken into the test set.", type=str)

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.latent_representations_folder != None and args.calibration_folder != None and args.test_folder != None and args.layer_indices != None and args.random_seed != None and args.test_proportion != None, "If no configuration file is provided, then all other arguments must be provided."
    
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        calibration_folder = json.loads(args.calibration_folder)
        calibration_folder_path = os.path.join(*calibration_folder)
        test_folder = json.loads(args.test_folder)
        test_folder_path = os.path.join(*test_folder)
        layer_indices = json.loads(args.layer_indices)
        random_seed = json.loads(args.random_seed)
        test_proportion = json.loads(args.test_proportion)
                
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.latent_representations_folder == None and args.calibration_folder == None and args.test_folder == None and args.layer_indices == None and args.random_seed == None and args.test_proportion == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'split_latent_yamnet_into_calibration_and_test_sets' or configuration['script'] == 'split_latent_yamnet_into_calibration_and_test_sets.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'split_latent_yamnet_into_calibration_and_test_sets'."
        
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        calibration_folder_path = os.path.join(*configuration['arguments']['calibration_folder'])
        test_folder_path = os.path.join(*configuration['arguments']['test_folder'])
        layer_indices = configuration['arguments']['layer_indices']
        random_seed = configuration['arguments']['random_seed']
        test_proportion = configuration['arguments']['test_proportion']
    
    print("\n\n\tStarting script split_latent_yamnet_into_calibration_and_test_sets")
    print("\t\tThe script parsed the following arguments:")
    print("\t\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\t\tcalibration_folder path: ", calibration_folder_path)
    print("\t\ttest_folder path: ", test_folder_path)
    print("\t\tlayer_indices: ", layer_indices)
    print("\t\trandom_seed: ", random_seed)
    print("\t\ttest_proportion: ", test_proportion)
    print("\n\tStarting script now:\n")

    ### Start actual data processing
    
    # File management
    assert not os.path.exists(calibration_folder_path), f"\t\tFound existing folder at {calibration_folder_path}. ABORTING SCRIPT."
    assert not os.path.exists(test_folder_path), f"\t\tFound existing folder at {test_folder}. ABORTING SCRIPT."
    os.makedirs(calibration_folder_path)
    os.makedirs(test_folder_path)
    
    # Ensure all files exist for all layers
    file_names = set(os.listdir(os.path.join(latent_representations_folder_path, f'Layer {layer_indices[0]}')))
    file_count = len(file_names)
    for layer_index in layer_indices[1:]: 
        current_file_names = set(os.listdir(os.path.join(latent_representations_folder_path, f'Layer {layer_index}')))
        assert len(file_names.intersection(current_file_names)) == file_count, f"\t\tNot all files are the same across layers. Layer {layer_index} has a distinct set of files than layer f{layer_indices[0]}. ABORTING SCRIPT."
    
    # Assert proportion
    assert type(test_proportion) == float and test_proportion > 0 and test_proportion < 1.0, f"The input test_proportion should be a float between 0 and 1."
    
    # Create list of indices for the two sets
    file_names = list(file_names)
    file_names.sort() # Needed, because otherwise the file_names are pre-shuffled in an unpredictable way
    random.seed(random_seed)
    random.shuffle(file_names)
    split_index = (int)(len(file_names)*(1-test_proportion))

    # Iterate layers
    for l, layer_index in enumerate(layer_indices):
        print(f"\n\t\tLayer {layer_index} ", end="")

        # Move files
        os.makedirs(os.path.join(calibration_folder_path, f'Layer {layer_index}'))
        for file_name in file_names[:split_index]:
            os.rename(os.path.join(latent_representations_folder_path, f'Layer {layer_index}', file_name), 
                      os.path.join(calibration_folder_path, f'Layer {layer_index}', file_name))
            
        os.makedirs(os.path.join(test_folder_path, f'Layer {layer_index}'))
        for file_name in file_names[split_index:]:
            os.rename(os.path.join(latent_representations_folder_path, f'Layer {layer_index}', file_name), 
                      os.path.join(test_folder_path, f'Layer {layer_index}', file_name))
            
        print(f"Completed")
        
    # Log
    print("\n\n\Completed script split_latent_yamnet_into_calibration_and_test_sets")
            