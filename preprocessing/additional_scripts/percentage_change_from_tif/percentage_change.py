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

#definiton of variables
source_folder = ""
target_folder = ""
files = []#List of lists; Each inner list contains the names of files to be treated as one
files_full = []

def load_frames(filepaths, debug = False):
    frames = []
    for filepath in filepaths:
        print(filepath.split("/")[-1])
        current_frames = skimage.io.imread(filepath)
        current_frames = current_frames.astype(np.double)
        frames.append(current_frames)
        if debug:
            break

    frames = np.concatenate(frames, axis=0)
    if debug:
        frames = frames[:3000]
    return frames

def percentage_change(frames, mask):
    mean = np.mean((frames+1), axis=0)
    percentage_change = ((frames+1) - mean)
    percentage_change /= mean
    percentage_change *= 100
    
    n_frames = len(percentage_change)
    ran = list(range(0,n_frames,1000))
    ran.append(n_frames)
    for start, stop in zip(ran[:-1],ran[1:]):
        smooth = gaussian_filter(percentage_change[start:stop], 2)
        smooth = remove_frequency_from_pixel_vectors(smooth, 10, 20, "fourier")
        percentage_change[start:stop] = smooth
        print(".", end = "")
        sys.stdout.flush()
    percentage_change = gaussian_filter(percentage_change, [1,0,0])
    percentage_change = apply_mask(percentage_change, mask, nan=True)
    
    return percentage_change    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    use_default_masks = False
    current_session = 0
    
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.rstrip('\n')
                if "#" in line:
                    continue
                files.append(line.split(" "))
                files_full.append([os.path.join(source_folder,f) for f in files[-1]])
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()

    #Check that files exist
    for names in files_full:
        for path in names:
          if not os.path.isfile(path):
              print(filepath +" is not a file")
              sys.exit()

    Path(target_folder).mkdir(parents=True, exist_ok=True)

    #Write some metainfo
    metainfo = {"source_folder": source_folder}
    info = json.dumps({**metainfo,**vars(args)}, indent=4)
    with open(os.path.join(target_folder,"metainfo.txt"), "w") as f:
        f.write(info)

    for filenames in files:
        inputs = [os.path.join(source_folder,name) for name in filenames]
        output = os.path.join(target_folder, os.path.basename(inputs[0]).split(".")[0])#e.g. runstart16
        mask = os.path.join(source_folder, inputs[0].split(".")[0]+"_mask.png")
        img = Image.open(os.path.join(source_folder,mask))
        mask = np.array(img)==255
        np.save(output, percentage_change(load_frames(inputs), mask))
