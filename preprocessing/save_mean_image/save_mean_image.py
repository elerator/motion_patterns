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

from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation

from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects

#definiton of variables
source_folder = ""
files = []

def preprocess(input, out, out1, mask_out, mask_only):
    frames = skimage.io.imread(filepath)
    frames = frames.astype(np.double)
    mean = np.mean(frames[:100],axis=0)#pixelwise mean
    mask = np.array((normalize(mean)>.2))
    
    mask = remove_small_holes(mask, 500)
    mask = gaussian_filter(mask*10, 1) > 1#make outline more round

    mask = binary_erosion(mask, iterations = 15)#shrik outline
    mask = binary_dilation(mask, iterations = 8)#shrink and smoothen outline
    mask = gaussian_filter(mask*10, 5) > 1#make outline more round

    mask = mask.astype(np.int32) * -1
    mask += 1
    mask = mask*255
    Image.fromarray(mask.astype(np.uint8)).save(mask_out)
    print("saved mask successfully")

    if mask_only:
        return

    Image.fromarray(np.array(normalize(mean)*255, dtype=np.uint8)).save(out1)

    difference = frames - mean
    difference = normalize(difference)
    difference = np.mean(difference, axis=0)
    difference = normalize(difference)
    difference = mask * 255
    difference = Image.fromarray(difference.astype(np.uint8))
    difference.save(out)

    #print(filepath.split("/")[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')
    parser.add_argument('--mask_only', action = "store_true")

    args = parser.parse_args()

    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
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

    for name in files:
        input = os.path.join(source_folder,name)
        pixelwise_mean_of_contrast = os.path.join(source_folder,os.path.splitext(name)[0]+"_mean_of_pixelwise_mean.png")
        mean_out = os.path.join(source_folder,os.path.splitext(name)[0]+"_pixelwise_mean.png")
        mask_out = os.path.join(source_folder,os.path.splitext(name)[0]+"_mask.png")
        preprocess(input, pixelwise_mean_of_contrast, mean_out, mask_out, args.mask_only)
