"""This script augments the raw audio data by applying reverberation.

Requirements:
- The audio data needs to be sampled at 48 Khz with int16 bit rate
- in addition to the packages installed with this python project, the instructions for installing pysndfx and its dependency sox need to be followed https://pypi.org/project/pysndfx/

Steps:
- Loads all sound files
- Applies the augmentation
- Saves the new sounds as new files 

Side effects:
if the script has been run before, the current new sound files will override the previous new ones.
"""


from latent_audio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .lowshelf()
)

# Configuration
raw_folder_path = os.path.join('data','raw audio')
augmented_folder_path = os.path.join('data','augmented audio')
if not os.path.exists(augmented_folder_path): os.makedirs(augmented_folder_path)

raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
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

    print(f'{100*c/len(raw_file_names)}% Finished')

print("Script completed")            


