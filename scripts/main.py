import sys 
sys.path.append(".")
import argparse, json
from LatentAudio.configurations import loader as configuration_loader

if __name__ == "__main__":
    # imports
    import argparse, os, json, sys

    # Parse input arguments
    parser = argparse.ArgumentParser(
        prog="main",
        description='''This script runs through the entire analysis, as specfiied by the provided configuration file. It is expected to be run using the command line tool from within a direct parent folder to the LatentAudio folder.''')

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    args = parser.parse_args()
    
    # Load configuration
    file_path = args.configuration_file_path
    configuration = configuration_loader.load_configuration(file_path=file_path)

    print(f"\n\nRunning main script with configuration: {configuration['name']}\n")

    # Run all scripts from configuration
    for s, step in enumerate(configuration["steps"]):
        # Name the script
        script_name = step["script"]
        if ".py" not in script_name: script_name += '.py'
        path_to_script = os.path.join('LatentAudio','scripts', script_name)
        
        # Collect its arguments
        tmp = [path_to_script]
        for key, value in step["arguments"].items():
            tmp.append("--" + key)
            tmp.append((str)(value))
        sys.argv = [script_name, '--configuration_file_path', file_path, '--configuration_step', str(s)]

        # Execute script
        exec(open(path_to_script).read())
        