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
import skimage
from skimage import io

import json

def percentage_change(frames, mask, smoothing = (2,10,10), debug = False):
    mean = np.mean((frames+1), axis=0)
    if debug:
       frames = frames[:2000]
    percentage_change = ((frames+1) - mean)
    percentage_change /= mean
    percentage_change *= 100
    
    n_frames = len(percentage_change)
    ran = list(range(0,n_frames,1000))
    ran.append(n_frames)
    for start, stop in zip(ran[:-1],ran[1:]):#process in chunks to limit amount of required memory
        smooth = gaussian_filter(percentage_change[start:stop], smoothing)
        smooth = remove_frequency_from_pixel_vectors(smooth, 10, 20, "fourier")
        percentage_change[start:stop] = smooth
        print(".", end = "")
        sys.stdout.flush()
    percentage_change = gaussian_filter(percentage_change, [1,0,0])
    if type(mask) != type(None):
    	percentage_change = apply_mask(percentage_change, mask, nan=True)
    
    return percentage_change

def get_filenames(args):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            mask_path = f.readline()[:-1]
            print("Using mask "+ mask_path)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    input_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    input_paths = [os.path.join(source_folder, f) for f in input_files]
    output_paths = [os.path.join(target_folder, f) for f in input_files]
    return source_folder, target_folder, mask_path, input_paths, output_paths

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    use_default_masks = False
    current_session = 0

    source_folder, target_folder, mask_path, input_files, output_files = get_filenames(args)
    output_files = remove_existing_files(output_files)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    #Write some metainfo
    #metainfo = {"source_folder": source_folder}
    #info = json.dumps({**metainfo,**vars(args)}, indent=4)
    #with open(os.path.join(target_folder,"metainfo.txt"), "w") as f:
    #    f.write(info)

    for input_file, output_file in zip(input_files, output_files):
        if not output_file:
           continue
        print(".", end="")
        img = Image.open(mask_path)
        mask = np.array(img)==255
        np.save(output_file, percentage_change(np.load(input_file), mask, debug = args.debug))
