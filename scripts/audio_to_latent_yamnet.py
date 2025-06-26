
import sys
sys.path.append(".")
import tensorflow as tf
from LatentAudio.adapters import layer_wise as ylw      
import argparse, json
from LatentAudio.configurations import loader as configuration_loader
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="audio_to_latent_yamnet",
        description='''This script loads a set of sound files from the given `sounds_folder` and passes each of them through the Yamnet convolutional neural network. 
                        For a given sound, Yamnet will create a spectrogram and split it into ca. 1-second long, partially ovelapping time-frames. Each time-frame is
                        fed through Yamnet up until a given layer index. Then `time_frames_per_slice` many such consecutive time_frames are put into a slice and then 
                        averaged across time. The start times of consecutive slices are `offset` many time-frames apart. The resulting latent representation of a given slice
                        will be extracted and saved to disk.
                        Saving takes place by moving any previous outputs located in the provided `latent_representations_folder` to a new folder whose name is extend by ' (old)' and a time-stamp.
                        Then, a new folder is created at `latent_representations_folder` and inside that folder, for each Yamnet layer, a subfolder named as the current 
                        Yamnet layer will be created. 
                        Inside that subfolder, each slice is saved with the file name <waveform file name>_X_<slice index>.npy. 
                        For example, if the waveform file for Metal Tapping was called 'MT.wav', the slice files would be called 'MT_X_1.npy', 'MT_X_2.npy', etc.
                        Furthermore, the script creates a bar-chart showing the original dimensionalities of the flattened latent representations. The figure will be
                        saved at `figure_folder` using the title 'Yamnet Dimensionalities'. If that file already exists, it will be renamed using the 
                        appendix ' (old) ' and a time-stamp before the new file is saved.

                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--sounds_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder containing the sound data in .wav format. This sound data needs to be sampled at 16 Khz with int16 bit rate.", type=str)
    parser.add_argument("--sound_file_names", help="A list of strings (in form of one json string) that contains the file_names of the sounds that shall be converted.", type=str)
    parser.add_argument("--latent_representations_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations shall be stored. If the folder does not yet exist, it will be created. If it already exists, it will not be replaced.", type=str)
    parser.add_argument("--layer_indices", help="A list (in form of one json string) containing the indices of the Yamnet layers for which the latent vectors of sounds shall be computed.", type=str)
    parser.add_argument("--time_frames_per_slice", help="An int (in form of one json string) indicating how many consecutive time-frames should be in a slice.", type=str)
    parser.add_argument("--offset", help="An int (in form of one json string) indicating how many time frames should be skipped before starting a new slice. Should be at least 1. If set to 1, then the starting times of two consecutive slices will be 1 time-frame apart. If set to 2, then 2 time frames apart, etc. Keep in mind that time-frames partially overlap by Yamnet's internal processing.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot of original dimensionalities should be saved.")

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.sounds_folder != None and args.sound_file_names != None and args.latent_representations_folder != None and args.layer_indices != None and args.time_frames_per_slice != None and args.offset != None and args.figure_folder != None, "If no configuration file is provided, then all other arguments must be provided."
    
        sounds_folder = json.loads(args.sounds_folder)
        sounds_folder_path = os.path.join(*sounds_folder)
        sound_file_names = json.loads(args.sound_file_names)
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        layer_indices = json.loads(args.layer_indices)
        time_frame_per_slice = json.loads(args.time_frames_per_slice)
        offset = json.loads(args.offset)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)

    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.sounds_folder == None and args.sound_file_names == None and args.latent_representations_folder == None and args.layer_indices == None and args.time_frames_per_slice == None and args.offset == None and args.figure_folder == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'audio_to_latent_yamnet' or configuration['script'] == 'audio_to_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equalt to 'audio_to_latent_yamnet'."
        
        sounds_folder_path = os.path.join(*configuration['arguments']['sounds_folder'])
        sound_file_names = configuration['arguments']['sound_file_names']
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        layer_indices = configuration['arguments']['layer_indices']
        time_frames_per_slice = configuration['arguments']['time_frames_per_slice']
        offset = configuration['arguments']['offset']
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])

    print("\n\n\tStarting script audio_to_latent_yamnet")
    print("\t\tScript parsed the following arguments:")
    print("\t\tsounds_folder path: ", sounds_folder_path)
    print("\t\tsound_file_names: ", sound_file_names)
    print("\t\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\t\tlayer_indices: ", layer_indices)
    print("\t\ttime_frames_per_slice: ", time_frames_per_slice)
    print("\t\toffset: ", offset)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\n\tStarting processing now:\n")

    ### Start actual data processing
    
    # Remove any previous outputs
    if os.path.exists(latent_representations_folder_path): 
        print(f"\t\tFound existing folder at {latent_representations_folder_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(latent_representations_folder_path,
                  latent_representations_folder_path + " (old) " + str(time.time()))
    
    # Convert audio to latent Yamnet
    dimensionalities = [None] * len(layer_indices)
    for l, layer_index in enumerate(layer_indices):
            
        # Prepare Yamnet
        tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
        yamnet = ylw.LayerWiseYamnet()
        yamnet.load_weights(os.path.join('LatentAudio','plugins','yamnet','yamnet.h5'))
        
        # Ensure the output folder exist
        layer_path = os.path.join(latent_representations_folder_path, f'Layer {layer_index}')
        os.makedirs(layer_path)

        # Log
        print(f"\n\t\tLayer {layer_index}.")
        
        print(f"\r\t\t\t0 % Completed", end='')
        for r, sound_file_name in enumerate(sound_file_names):

            # Load .wav file
            waveform, sampling_rate = sf.read(os.path.join(sounds_folder_path,sound_file_name), dtype=np.int16)
            assert sampling_rate == 16000, f"The sampling rate of the sound file {sound_file_name} was assumed to be 16000 in order to apply the decimation algorithm and achieve Yamnet's 16000. The provided audio has sampling rate {sampling_rate}. You need to use a different downsampling method, e.g. from sklearn to meet Yamnet's requirement."
            
            # Normalize
            waveform =  waveform.astype('float32')
            waveform = waveform / np.max(waveform)
            
            # Pass through yamnet up until target layer
            latent = yamnet.call_until_layer(waveform=waveform, layer_index=layer_index).numpy()
            dimensionalities[l] = np.product(latent.shape[1:]) # Flattened dimensionality
            
            # Slice along time axis into slices of length time_frames_per_slice with offset
            i = 0
            for start in range(0, latent.shape[0]-time_frames_per_slice+1, offset):
                stop = start + time_frames_per_slice
                current_slice = latent[start:stop,:]

                # Average each slice across time (axis 0)
                current_slice = np.mean(current_slice, axis=0)

                # Save
                np.save(os.path.join(layer_path, f"{sound_file_name[:-4]}_X_{i}.npy"), np.reshape(current_slice,[-1]))
                i += 1

            # Log
            print(f"\r\t\t\t{np.round(100*(r+1)/len(sound_file_names))} % Completed", end='')
            
            # Delete singleton
            del latent  
        
        del yamnet
    
    # Plot original dimensionalities of Yamnet
    print("\t\tCreating figure for Yamnet dimensionalities now")
    plt.figure(figsize=(len(layer_indices),5)); plt.title("Yamnet Dimensionalities")
    plt.bar([f'{key}' for key in layer_indices], dimensionalities, color='white', edgecolor='black')
    plt.ylabel("Dimensionality"); plt.xlabel('Layer')
    
    # Save figure
    if not os.path.exists(figure_folder_path): os.makedirs(figure_folder_path)
    figure_path = os.path.join(figure_folder_path, "Yamnet Original Layer-wise Dimensionalities")
    if os.path.exists(figure_path): 
        print(f"\t\tFound existing figure at {figure_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(figure_path, figure_path + ' (old) ' + (str)(time.time()))
    plt.tight_layout()
    plt.savefig(figure_path)

    # Log
    print("\n\n\Completed script audio_to_latent_yamnet")