import sys
sys.path.append(".")
import argparse, json
from LatentAudio.adapters import layer_wise as ylw
from LatentAudio.configurations import loader as configuration_loader
import os, soundfile as sf, numpy as np, shutil
from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .lowshelf()
)

if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="augment_audio",
        description='''This script loads all sound files at `raw_sounds_folder`, augments the them by applying reverberation and saves them one by one to `augmented_sounds_folder`. 
                    The audio files stored at `raw_sounds_folder` are assumed to be .wav files recorded with an int16 bitrate. If there are any existing augmented sound files of same name as the new ones, they will be replaced by the new ones.
                        
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--raw_sounds_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder containing the raw sound data.", type=str)
    parser.add_argument("--augmented_sounds_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the augmented sounds shall be stored. If the folder does not yet exist, it will be created. If it already exists, it will not be replaced.", type=str)
    parser.add_argument("--file_name_suffix", help='A string appended to each file-name to indicate that this file contains the augmented sound.', type=str)
    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.raw_sounds_folder != None and args.augmented_sounds_folder != None and args.file_name_suffix != None, "If no configuration file is provided, then all other arguments must be provided."
    
        raw_sounds_folder = json.loads(args.raw_sounds_folder)
        raw_sounds_folder_path = os.path.join(*raw_sounds_folder)
        augmented_sounds_folder = json.loads(args.augmented_sounds_folder)
        augmented_sounds_folder_path = os.path.join(*augmented_sounds_folder)
        file_name_suffix = args.file_name_suffix
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.raw_sounds_folder == None and args.augmented_sounds_folder == None and args.file_name_suffix == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'augment_audio' or configuration['script'] == 'augment_audio.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'augment_audio'."
        
        raw_sounds_folder_path = os.path.join(*configuration['arguments']['raw_sounds_folder'])
        augmented_sounds_folder_path = os.path.join(*configuration['arguments']['augmented_sounds_folder'])
        file_name_suffix = args.file_name_suffix
        
    print("\n\nThe script augment_audio parsed the following arguments:")
    print("\traw_sounds_folder path: ", raw_sounds_folder_path)
    print("\taugmented_sounds_folder path: ", augmented_sounds_folder_path)
    print("\tfile_name_suffix: " + file_name_suffix)
    print("Starting script now:\n")

    ### Start actual data processing
    # Ensure output folder exists
    if not os.path.exists(augmented_sounds_folder_path): os.makedirs(augmented_sounds_folder_path)

    # Get list of input wav files
    raw_file_names = os.listdir(raw_sounds_folder_path) 
    for file_name in reversed(raw_file_names):
        if '.wav' not in file_name: raw_file_names.remove(file_name)

    # Preprocess all files
    for c, raw_file_name in enumerate(raw_file_names):

        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_sounds_folder_path, raw_file_name), dtype=np.int16)
        
        # Apply effects
        waveform = fx(waveform)
        assert waveform.dtype == np.int16

        # Save
        sf.write(os.path.join(augmented_sounds_folder_path, raw_file_name[-4:] + file_name_suffix + '.wav'), waveform, sampling_rate)

        print(f"\r\t{np.round(100*(c+1)/len(raw_file_names))}% Completed", end='')

    print("Script completed")      