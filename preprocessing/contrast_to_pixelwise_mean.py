import sys
sys.path.append('..')

#import of utilities
from utils.visualization_tools import *
import utils.visualization_tools
from utils.data_transformations import *
import utils.data_transformations
from utils.diverse import *
import utils.diverse

#imports
from pathlib import Path
import argparse
import skimage
from skimage import io

from scipy.ndimage.morphology import grey_closing
import json

#definiton of variables
source_folder = ""
target_folder = ""
files = []

def preprocess(filepath, output, debug = True, butter_not_fourier=False, remove_frequency = True):
    print(filepath.split("/")[-1])
    #frames = np.array(skimage.io.imread(os.path.join(filepath)), dtype=np.double)
    frames = skimage.io.imread(filepath)
    frames = frames.astype(np.double)

    if butter_not_fourier:
        fourier_not_butter = "butter"
    else:
        fourier_not_butter = "fourier"

    if debug:
        frames = frames[:1000]

    print("Processing tensor with " + str(len(frames)) + " frames")
    mean = np.mean(frames,axis=0)#pixelwise mean
    difference = framewise_difference(frames, mean, bigdata=True)

    #difference = normalize(difference)

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

    np.save(output, difference)
    print("saved successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--butter_not_fourier', action="store_true")
    parser.add_argument('--remove_frequency', action="store_true")

    args = parser.parse_args()

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
        out = os.path.join(target_folder,os.path.splitext(name)[0])
        preprocess(input, out, args.debug, args.butter_not_fourier, args.remove_frequency)
