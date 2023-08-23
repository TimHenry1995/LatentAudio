import os, soundfile as sf, numpy as np, shutil
"""This script concatenates the narration and sounds into an audioguide. It assumes that all recordings have the same sampling rate.
"""

# Configuration
slice_folder_path = os.path.join('data','sound slices for humans')
narration_folder_path = os.path.join('data','narration')

block_count = 4

audio_guide, sampling_rate = sf.read(os.path.join(narration_folder_path,'Onboarding.wav'))

# Concatenate
for b in range(block_count):
    
    block_narration, _ = sf.read(os.path.join(narration_folder_path, f'BLOCK_{b}.wav')) 
    audio_guide = np.concatenate([audio_guide, block_narration,0*block_narration]) # Small break to let participant rest
    max = np.max(audio_guide)
    if 1 < max: audio_guide = audio_guide / max
        
    # Trials
    block_path = os.path.join(slice_folder_path,f"Block {b}")
    trial_file_names = os.listdir(block_path)
    for name in reversed(trial_file_names):
        if '.wav' not in name: trial_file_names.remove(name)
    
    # Sort trials
    tmp = [None] * len(trial_file_names)
    
    for name in trial_file_names:
        i = (int)(name[:2]) # The first two characters contain the index
        tmp[i] = name
    trial_file_names = tmp; del tmp

    for t, trial_file_name in enumerate(trial_file_names):
        
        trial_narration, _ = sf.read(os.path.join(narration_folder_path, f'TRIAL_{t}.wav'))
        max = np.max(trial_narration)
        if 1 < max: trial_narration = trial_narration / max
        
        trial_sound, _ = sf.read(os.path.join(block_path, trial_file_name))
        max = np.max(trial_sound)
        if 1 < max: trial_sound = trial_sound / max
        audio_guide = np.concatenate([audio_guide, trial_narration, trial_sound])

# Save
sf.write(os.path.join(narration_folder_path, "Audio Guide.wav"), audio_guide, samplerate=sampling_rate)