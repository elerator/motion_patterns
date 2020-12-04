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
from collections import defaultdict

def get_filenames(args):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            dataset_path = f.readline()[:-1]
            print("Using target_folder " + dataset_path)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    input_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    input_paths = [os.path.join(source_folder, f) for f in input_files]
    return source_folder, dataset_path, input_paths

def add_feature(dataset, id, tensor):
    dataset["sws"][id]["gcamp_improved_interpolated"] = np.nanmean(tensor, axis = (1,2))
    dataset["sws"][id]["gcamp_improved_interpolated"] = stretch(normalize(np.nanmean(tensor, axis = (1,2))), 128)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--override')
    args = parser.parse_args()
    use_default_masks = False

    source_folder, dataset_path, input_files = get_filenames(args)

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    for i, input_file in enumerate(input_files):
        print(str(100*i/len(input_files))+"%")
        dataset = add_feature(dataset, os.path.basename(input_file).split(".")[0], np.load(input_file))
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
