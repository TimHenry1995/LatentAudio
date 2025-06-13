import sys
sys.path.append(".")
from LatentAudio.adapters import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .lowshelf()
)
def run(raw_audio_data_folder = os.path.join('LatentAudio','data','raw audio'), augmented_audio_data_folder = os.path.join('LatentAudio','data','augmented audio')) -> None:
    """This function loads all sound files at `raw_audio_data_folder`, augments the raw audio data by applying reverberation and saves them one by one to `augmented_audio_data_folder`. 
    The audio files stored at `raw_audio_data_folder` are assumed to be .wav files recorded with an int16 bitrate. 

    :param raw_audio_data_folder: Path to the folder containing the raw audio files.
    :type raw_audio_data_folder: str
    :param augmented_audio_data_folder: Path to the folder where the augmented audio files will be saved.
    :type augmented_audio_data_folder: str
    :return: None
    :rtype: NoneType
    """
    
    # Ensure output folder exists
    if not os.path.exists(augmented_audio_data_folder): os.makedirs(augmented_audio_data_folder)

    # Get list of input wav files
    # Assumes files to have material as first letter and action as second letter
    raw_file_names = os.listdir(raw_audio_data_folder) 
    for file_name in reversed(raw_file_names):
        if '.wav' not in file_name: raw_file_names.remove(file_name)

    # Preprocess all files
    for c, raw_file_name in enumerate(raw_file_names):

        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_audio_data_folder, raw_file_name), dtype=np.int16)
        
        # Apply effects
        waveform = fx(waveform)
        assert waveform.dtype == np.int16

        # Save
        sf.write(os.path.join(augmented_audio_data_folder, raw_file_name), waveform, sampling_rate)

        print(f"\r\t{np.round(100*(c+1)/len(raw_file_names))}% Completed", end='')

    print("Script completed")      

if __name__ == "__main__":
    # Load Configuration
    import json, os
    with open(os.path.join('LatentAudio','configuration.json'),'r') as f:
        configuration = json.load(f)

    # Augment audio
    run(raw_audio_data_folder = configuration['raw_audio_data_folder'],
        augmented_audio_data_folder = configuration['augmented_audio_data_folder'])