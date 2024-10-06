

from latent_audio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .lowshelf()
)
def run(raw_folder_path = os.path.join('data','raw audio'), augmented_folder_path = os.path.join('data','augmented audio')) -> None:
    """This function loads all sound files at `raw_folder_path`, augments the raw audio data by applying reverberation and saves them one by one to `augmented_folder_path`. 
    The audio files stored at `raw_folder_path` are assumed to be .wav files recorded with an int16 bitrate. 

    :param raw_folder_path: Path to the folder containing the raw audio files.
    :type raw_folder_path: str
    :param augmented_folder_path: Path to the folder where the augmented audio files will be saved.
    :type augmented_folder_path: str
    :return: None
    :rtype: NoneType
    """
    
    # Ensure output folder exists
    if not os.path.exists(augmented_folder_path): os.makedirs(augmented_folder_path)

    # Get list of input wav files
    # Assumes files to have material as first letter and action as second letter
    raw_file_names = os.listdir(raw_folder_path) 
    for file_name in reversed(raw_file_names):
        if '.wav' not in file_name: raw_file_names.remove(file_name)

    # Preprocess all files
    for c, raw_file_name in enumerate(raw_file_names):

        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        
        # Apply effects
        waveform = fx(waveform)
        assert waveform.dtype == np.int16

        # Save
        sf.write(os.path.join(augmented_folder_path, raw_file_name), waveform, sampling_rate)

        print(f"\r\t{np.round(100*c/len(raw_file_names))}% Completed", end='')

    print("Script completed")      
