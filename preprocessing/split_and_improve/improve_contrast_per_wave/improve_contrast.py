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

from skimage.filters.rank import entropy
from skimage.morphology import disk

import warnings
warnings.filterwarnings("ignore")

def improve_contrast(t):
    """ Improve contrast by computing percentage change for slow wave and substracting its mean """
    bg = normalize_nan(np.nanmean(t, axis=0))+1
    pc = t - bg / bg
    pc -= np.mean(pc, axis=0)
    return pc

def adaptive_smoothing(pc1, target_mean_entropy = .25, stepsize = 10, max_iterations = 15):
    """ Applies smoothing iteratively. Breaks if the mean local entropy falls below target_mean_entropy or the current iteration 
        exceeds max_iterations
    Args:
        pc1: percentage change input 3D tesor
        target_mean_entropy: Mean entropy for which no further smoothing is desired
        stepsize: Steps for subsampling slices
        max_iterations: Maximal number of iterations
    Returns:
        smoothed tensor with a decreased mean local entropy
    """
    print("adaptsmooth") 
    for _ in range(max_iterations):
        subvolume = pc1[::stepsize]
        mean_local_entropy = np.sum([np.sum(entropy(p, disk(3))) for p in subvolume])/len(subvolume.flatten())
        print(mean_local_entropy)
        if mean_local_entropy < target_mean_entropy:
            print("break")
            break
        pc1 = gaussian_filter_nan(pc1, [0, 5,5])
        
    return pc1

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

def process(input_file, output_file, adapt_smooth, debug):
    t = np.load(input_file)
    t = improve_contrast(t)
    if adapt_smooth:
       t = adaptive_smoothing(t)
    else:
       t = gaussian_filter_nan(t, [0,5,5])
       t = gaussian_filter_nan(t, [0,5,5])
    np.save(output_file, t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--override', action = "store_true")
    parser.add_argument('--adaptive_smoothing', action = "store_true")
    args = parser.parse_args()
    use_default_masks = False

    source_folder, target_folder, input_files, output_files = get_filenames(args)
    if not args.override:
    	output_files = remove_existing_files(output_files)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    n_files = len(output_files)
    for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
        if not output_file:
           print("File exists already. Use --override if you wish to recompute.")
           print("Skipping " + input_file)
           continue
        print(str(100*i/n_files)+"%")
        process(input_file, output_file, args.adaptive_smoothing, args.debug)
        if args.debug:
           break
