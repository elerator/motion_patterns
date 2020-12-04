import sys
sys.path.insert(0,'../../..')

import gc

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pathlib import Path
import argparse
import pickle


def get_filenames(args):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            tensor_folder = f.readline()[:-1]
            print("Using data from " + tensor_folder)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    return source_folder, target_folder, tensor_folder

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

def process(tensor, output_file, start, stop, negative_component = False, positive_component = False, negate = False):
    if negative_component:
       tensor[start:stop][tensor[start:stop] > 0] = 0
       #tensor[~np.isnan(tensor)] *= -1
    if positive_component:
       tensor[start:stop][tensor[start:stop] < 0] = 0
    if negate:
       tensor[start:stop][~np.isnan(tensor[start:stop])] = -tensor[start:stop][~np.isnan(tensor[start:stop])]
    np.save(output_file, tensor[start:stop])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--override')
    parser.add_argument('--negative_component', action="store_true")
    parser.add_argument('--negate', action="store_true")
    parser.add_argument('--continuous', action="store_true")
    parser.add_argument('--positive_component', action="store_true")
    args = parser.parse_args()
    use_default_masks = False

    input_folder, output_folder, tensor_folder = get_filenames(args)
    with open(os.path.join(input_folder, "dataset.pkl"), "rb") as f:
         dataset = pickle.load(f)
    ids = list(dataset["sws"].keys())
    input_files = [os.path.join(tensor_folder, dataset["sws"][f]["file_id"]+".npy") for f in ids]
    output_files = [os.path.join(output_folder, f + ".npy") for f in ids]
    
    if args.continuous:
       starts = [dataset["sws"][id]["start"] for id in ids]
       stops = [dataset["sws"][id]["stop"] for id in ids]
    else:
       starts = [dataset["sws"][id]["nc_start"] for id in ids]
       stops = [dataset["sws"][id]["nc_stop"] for id in ids]

    if not args.override:
    	output_files = remove_existing_files(output_files)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    prev_input_file = input_files[0]
    t = np.load(prev_input_file)
    for input_file, output_file, start, stop in zip(input_files, output_files, starts, stops):
           
        if not output_file:
           print("File exists already. Use --override if you wish to recompute.")
           print("Skipping " + input_file)
           continue
        
        if input_file != prev_input_file:
           prev_input_file = input_file
           t = np.load(input_file)
           
        print(".", end="")
        process(t, output_file, start, stop, args.negative_component, args.positive_component, args.negate)
