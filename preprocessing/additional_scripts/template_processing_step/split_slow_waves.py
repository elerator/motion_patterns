import sys
sys.path.insert(0,'../..')

import gc

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pathlib import Path
import argparse



def get_filenames(args):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    input_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    input_paths = [os.path.join(source_folder, f) for f in input_files]
    output_paths = [os.path.join(target_folder, f) for f in input_files]
    return source_folder, target_folder, input_paths, output_paths

def remove_existing_files(filepaths):
    """ Replaces files that already exist by None.
    Args:
        filepaths: List of strings representing filepaths
    Returns:
        List of filepaths where invalid files are None
    """
    valid = []
    for f in filepaths:
         if os.path.isfile(f):
            print(f + " exists already", end = " ")
            print("Delete file if you would like to recompute\n")
            valid.append(None)
         else:
            valid.append(f)
    return valid

def process(input_file, output_file, debug):
    print(input_file)
    print(output_file)
    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--override', action = "store_true")
    args = parser.parse_args()
    use_default_masks = False

    source_folder, target_folder, input_files, output_files = get_filenames(args)
    if not args.override:
    	output_files = remove_existing_files(output_files)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    for input_file, output_file in zip(input_files, output_files):
        if not output_file:
           print("File exists already. Use --override if you wish to recompute.")
           print("Skipping " + input_file)
           continue
        print(".", end="")
        process(input_file, output_file, args.debug)
