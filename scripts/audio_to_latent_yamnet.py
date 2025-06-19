
import sys
sys.path.append(".")
import argparse, json
from LatentAudio.configurations import loader as configuration_loader
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate
from typing import Dict, List

def run_at_layer(
    layer_index,
    sounds_folder_path,
    latent_representations_folder_path,
    ) -> None:

    # Prepare Yamnet
    import tensorflow as tf
    from LatentAudio.adapters import layer_wise as ylw
    tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
    yamnet = ylw.LayerWiseYamnet()
    yamnet.load_weights(os.path.join('LatentAudio','plugins','yamnet','yamnet.h5'))
    
    # Ensure the output folders exists
    layer_path = os.path.join(latent_representations_folder_path, f'Layer {layer_index}')
    if not os.path.exists(layer_path): os.makedirs(layer_path)

    # Log
    print(f"\nConverting sounds to latent Yamnet representations at layer {layer_index}.")
    
    # Compile list of .wav filenames
    file_names = os.listdir(sounds_folder_path)
    for file_name in reversed(file_names):
        if not '.wav' in file_name: file_names.remove(file_name)

    print(f"\r\t0 % Completed", end='')
    for r, sound_file_name in enumerate(file_names):
        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(sounds_folder_path,sound_file_name), dtype=np.int16)
        assert sampling_rate == 48000, f"The sampling rate of the sound file {sound_file_name} was assumed to be 48000 in order to apply the decimation algorithm and achieve Yamnet's 16000. The provided audio has sampling rate {sampling_rate}. You need to use a different downsampling method, e.g. from sklearn to meet Yamnet's requirement."
        
        # Normalize
        waveform =  waveform.astype('float32')
        waveform = waveform / np.max(waveform)
        
        # Down-sample to yamnet's 16000
        waveform = decimate(waveform, 3)
        sampling_rate = 16000
        
        # Pass through yamnet up until target layer
        latent = yamnet.call_until_layer(waveform=waveform, layer_index=layer_index).numpy()

        # Save
        for s, slice in enumerate(latent):
            np.save(os.path.join(layer_path, f"{sound_file_name[:-4]}_X_{s}.npy"), np.reshape(slice,[-1]))
        print(f"\r\t{np.round(100*(r+1)/len(file_names))} % Completed", end='')

        # Delete singleton
        del latent

if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="audio_to_latent_yamnet",
        description='''This script loads a set of sound files from the given sounds_folder and passes each of them through the Yamnet convolutional neural network. 
                        For a given sound, Yamnet will create a spectrogram and split it into ca. 1-second long, partially ovelapping slices. 
                        Each slice is then passed through Yamnet's convolutional layers. 
                        At each of the here specified layer_indices, the latent representation of a given slice will be extracted and saved to disk.
                        Saving takes place in the provided latent_representations_folder by first creating a subfolder named as the current Yamnet layer (if it does not already exist). 
                        Inside that subfolder, each slice is saved with the file name <waveform file name>_X_<slice index>.npy.
                        For example, if the waveform file for Metal Tapping was called 'MT.wav', the slice files would be called 'MT_X_1.npy', 'MT_X_2.npy', etc.
                        
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--sounds_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder containing the sound data in .wav format. This sound data needs to be sampled at 48 Khz with int16 bit rate.", type=str)
    parser.add_argument("--latent_representations_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations shall be stored. If the folder does not yet exist, it will be created. If it already exists, it will not be replaced.", type=str)
    parser.add_argument("--layer_indices", help="A list containing the indices of the Yamnet layers for which the latent vectors of sounds shall be computed.", type=str)
    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.sounds_folder != None and args.latent_representations_folder != None and args.layer_indices != None, "If no configuration file is provided, then all other arguments must be provided."
    
        sounds_folder = json.loads(args.sounds_folder)
        sounds_folder_path = os.path.join(*sounds_folder)
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        layer_indices = json.loads(args.layer_indices)

    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.sounds_folder == None and args.latent_representations_folder == None and args.layer_indices == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'audio_to_latent_yamnet' or configuration['script'] == 'audio_to_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equalt to 'audio_to_latent_yamnet'."
        
        sounds_folder_path = os.path.join(*configuration['arguments']['sounds_folder'])
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        layer_indices = configuration['arguments']['layer_indices']

    print("\n\nThe script audio_to_latent_yamnet parsed the following arguments:")
    print("\tsounds_folder path: ", sounds_folder_path)
    print("\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\tlayer_indices: ", layer_indices)
    print("Starting script now:\n")

    ### Start actual data processing

    # Convert audio to latent Yamnet
    for layer_index in layer_indices:
        run_at_layer(
            layer_index=layer_index,
            sounds_folder_path = sounds_folder_path, 
            latent_representations_folder_path = latent_representations_folder_path)   