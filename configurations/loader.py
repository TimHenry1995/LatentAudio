import json
import os
from typing import List

CONFIGURATION_FILE_SPECIFICATION = '''
The configuration file is assumed to be a json file that has a first level key "name" providing the name of the configuration and a first level key "steps" that provides a list of distionaries. 
Each such dictionary is assumed to have a key "script" providing the script name and a key "arguments" containing another dictionary with key-value pairs for the arguments of that script.'''

def verify_configuration(file_path: str):
    
    # Assertions
    assert type(file_path) == str, "The configuration file path is assumed to be a string."
    
    # Ensure path has file extension
    if file_path[-5:] != '.json': file_path += '.json'

    # Load configuration
    with open(file_path,'r') as f:
        configuration = json.load(f)

    # Assertions for name
    assert type(configuration) == type(dict({})), 'The configuration is not a dictionary. ' + CONFIGURATION_FILE_SPECIFICATION
    assert "name" in configuration.keys(), 'The configuration is missing the first level entry "name". ' + CONFIGURATION_FILE_SPECIFICATION
    
    # Assertions for steps
    assert "steps" in configuration.keys(), 'The configuration is missing the first level entry "steps". ' + CONFIGURATION_FILE_SPECIFICATION
    assert type(configuration["steps"]) == type([]), 'The configuration does not store a list at the first level entry "steps". ' + CONFIGURATION_FILE_SPECIFICATION
    for s, step in enumerate(configuration["steps"]):
        assert "script" in step.keys(), f'Configuration step {s} does not have an entry "script". ' + CONFIGURATION_FILE_SPECIFICATION
        if step["script"][-3:] == ".py": step["script"] = step["script"][:-3] 
        script_options = ["audio_to_latent_yamnet","augment_audio","create_scalers_and_PCA_model_for_latent_yamnet","disentangle","evaluate_disentangle","explore_latent_yamnet","latent_yamnet_to_calibration_data_set"]
        assert step["script"] in script_options, f"Configuration step {s} pertains to script {step['script']} but this script is not an option. Valid options are {script_options}."
        assert "arguments" in step.keys(), f'Configuration step {s} does not have an entry "arguments". ' + CONFIGURATION_FILE_SPECIFICATION
        # The actual arguments are not checked here. This is left for the scripts themselves.

def load_configuration_step(file_path: str, step: int | str): 
    f"""Loads the given 'step' from the configuration located at `file_path`.
    {CONFIGURATION_FILE_SPECIFICATION}."""

    # Assertions
    assert (type(step) == int or type(step) == str) and step >= 0, "The step is assumed to be an integer indexing the analysis step in the steps list of the configuration file."

    # Load configuration
    configuration = load_configuration(file_path=file_path)

    # Extract step
    steps = configuration['steps']
    step = steps[step]

    # Return
    return step

def load_configuration(file_path: str):
    f"""Loads the configuration located at `file_path`.
    {CONFIGURATION_FILE_SPECIFICATION}."""

    # Assertions
    assert type(file_path) == str, "The configuration file path is assumed to be a string."
    
    # Ensure configuration is proper
    verify_configuration(file_path=file_path)

    # Ensure path has file extension
    if file_path[-5:] != '.json': file_path += '.json'

    # Load configuration
    with open(file_path,'r') as f:
        configuration = json.load(f)

    # Return
    return configuration
    