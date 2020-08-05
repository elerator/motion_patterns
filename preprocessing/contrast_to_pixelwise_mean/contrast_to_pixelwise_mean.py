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
files = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(filepath, mask, output, mean_out, debug = True, butter_not_fourier=False,
                    remove_frequency = True, apply_sigmoid=False, improved_bg = False, devide_by_background=False, bg_indices = None):
    print(filepath.split("/")[-1])
    frames = skimage.io.imread(filepath)
    frames = frames.astype(np.double)

    if butter_not_fourier:
        fourier_not_butter = "butter"
    else:
        fourier_not_butter = "fourier"

    if debug:
        frames = frames[:3000]

    print("Processing tensor with " + str(len(frames)) + " frames")
    if type(bg_indices) == type(None):
        mean = np.mean(frames,axis=0)#pixelwise mean
    else:
        mean = np.mean(frames[bg_indices],axis=0)
    difference = framewise_difference(frames, mean, bigdata=True)

    print(".", end = "")
    sys.stdout.flush()

    n_frames = len(frames)
    ran = list(range(0,n_frames,1000))
    ran.append(n_frames)

    for start, stop in zip(ran[:-1],ran[1:]):
        smooth = gaussian_filter(difference[start:stop], 2)
        if remove_frequency:
            smooth = remove_frequency_from_pixel_vectors(smooth, 10, 20, fourier_not_butter)
        difference[start:stop] = smooth
        print(".", end = "")
        sys.stdout.flush()

    difference = gaussian_filter(difference, [1,0,0])

    if improved_bg:#Contrast to down state
        vector = np.mean(difference, axis=(1,2))
        down_state = np.zeros(len(vector))
        down_state[vector<np.percentile(vector, 25)] = 1
        print("repeat with down state periods as background")
        sys.stdout.flush()

        #Call rekursively with down states
        difference = None#Free memory
        gc.collect()#Manually invoke garbage collector
        preprocess(filepath, mask, output, mean_out, debug, butter_not_fourier,
                    remove_frequency, apply_sigmoid, improved_bg=False, devide_by_background=devide_by_background, bg_indices = np.array(down_state, dtype=np.bool))
        return

    if mask:
        mask = np.array(Image.open(mask))[:,:,0] == 255
        difference = apply_mask(difference, mask, nan=True)
    if apply_sigmoid:
        difference = sigmoid(difference/100)
    if devide_by_background:
        bg = np.nanmean(difference,axis=0)
        difference /= bg

    np.save(output, difference)
    np.save(mean_out, np.nanmean(difference, axis=(1,2)))

    print("saved successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--butter_not_fourier', action="store_true")
    parser.add_argument('--remove_frequency', action="store_true")
    parser.add_argument('--sigmoid', action = "store_true")
    parser.add_argument('--improved_bg', action = "store_true")
    parser.add_argument('--devide_by_background', action = "store_true")

    args = parser.parse_args()

    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            mask = f.readline()[:-1]
            print("Using mask " + mask)
            if mask == "None":
                mask = None
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.rstrip('\n')
                if "#" in line:
                    continue
                files.append(line)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()

    for name in files:
        filepath = os.path.join(source_folder,name)
        if not os.path.isfile(filepath):
            print(filepath +" is not a file")
            sys.exit()

    Path(target_folder).mkdir(parents=True, exist_ok=True)

    #Write some metainfo
    metainfo = {"source_folder": source_folder}
    info = json.dumps({**metainfo,**vars(args)}, indent=4)
    with open(os.path.join(target_folder,"metainfo.txt"), "w") as f:
        f.write(info)

    for name in files:
        input = os.path.join(source_folder,name)
        out = os.path.join(target_folder,os.path.splitext(name)[0]+".npy")
        mean_out = os.path.join(target_folder,os.path.splitext(name)[0]+"_mean.npy")
        preprocess(input, mask, out, mean_out, args.debug, args.butter_not_fourier, args.remove_frequency, args.sigmoid, args.improved_bg, args.devide_by_background)