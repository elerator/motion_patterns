import sys
sys.path.append('..')

#import of utilities
from utils.data_transformations import *
import utils.data_transformations

#imports
from pathlib import Path
import argparse
import skimage
from skimage import io
from PIL import Image

#definiton of variables
source_folder = ""
target_folder = ""
files = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def smooth_binarization(frames, k = 10):
    #frames = (frames * k)-(k/2)
    frames = np.array([sigmoid(f) for f in frames])
    return frames

def preprocess(filepath, expected, output, debug = True, make_expected_mean_match = False, median=False, mask=None):
    print(filepath.split("/")[-1])
    if ".tif" in filepath:#Do preprocessing all in one
        frames = np.array(skimage.io.imread(os.path.join(filepath)), dtype=np.double)
        if debug:
            frames = frames[:2000]
        print("Processing tensor with " + str(len(frames)) + " frames")
        mean = np.mean(frames,axis=0)#pixelwise mean
        print(".", end = "")
        sys.stdout.flush()
        difference = framewise_difference(frames, mean, bigdata=True)
        mean = None
    elif ".npy" in filepath:#Use precomputed contrast_to_pixelwise_mean
        difference = np.load(filepath)

    print(".", end = "")
    sys.stdout.flush()
    expected = np.load(expected)
    expected = gaussian_filter(expected,2)
    if type(mask) != type(None):
        mask = np.array(Image.open(mask))[:,:,0] == 255
        expected = apply_mask(expected, mask, nan=True)

    if median:
        expected_mean = np.nanmedian(expected, axis=(1,2))
    else:
        expected_mean = np.nanmean(expected, axis=(1,2))


    n_frames = len(difference)
    ran = list(range(0,n_frames,1000))
    ran.append(n_frames)
    for start, stop in zip(ran[:-1],ran[1:]):
        print(start)
        print(stop)
        #frames = difference[start:stop].copy()
        frames = difference[start:stop]
        print(".", end = "")
        sys.stdout.flush()
        #frames = remove_frequency_from_pixel_vectors(frames,15,20)
        print(".", end = "")
        sys.stdout.flush()
        #frames = gaussian_filter(frames,2)
        if ".tif" in filepath:
            frames = gaussian_filter(frames,2)#TODO smoothing issue?

        if median:
            mean = np.nanmedian(frames,axis=(1,2))
        else:
            mean = np.nanmean(frames,axis=(1,2))#pixelwise mean

        column_vectors = np.tile(np.array([expected_mean]).transpose(), (1, len(mean)))
        mapping = np.argmin(np.abs(column_vectors - mean), axis=0)
        for i, expected_image_idx in enumerate(mapping):
            closest_matching_expectation = expected[expected_image_idx].copy()
            #print(np.nanmean(frames[i])-np.nanmean(expected[expected_image_idx]))
            print(expected_image_idx)

            if make_expected_mean_match:
                factor = np.nanmean(frames[i])/np.nanmean(closest_matching_expectation)
                closest_matching_expectation *= factor
            frames[i] -= closest_matching_expectation

        print(".", end = "")
        sys.stdout.flush()
        difference[start:stop] = frames

    print(np.max(difference))
    print(np.min(difference))
    #difference -= np.min(difference)
    difference /= 50
    #difference = normalize(difference)
    difference = smooth_binarization(difference)#TODO solve normalization issue
    np.save(output, difference)
    print("saved successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')
    parser.add_argument('--make_expected_mean_match', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--median', action="store_true")
    parser.add_argument('--mask', action="store_true")

    args = parser.parse_args()

    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            expected_images_folder = f.readline()[:-1]
            print("Using expected images folder " + expected_images_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)


            mask = f.readline()[:-1]
            print("Using mask " + mask)
            if not args.mask:
                mask = None
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
        filepath = os.path.join(expected_images_folder,os.path.splitext(name)[0]+"_expected.npy")
        if not os.path.isfile(filepath):
            print(filepath +" is not a file")
            sys.exit()

    for name in files:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        input = os.path.join(source_folder,name)
        expected =os.path.join(expected_images_folder,os.path.splitext(name)[0]+"_expected.npy")
        out = os.path.join(target_folder,os.path.splitext(name)[0])
        preprocess(input, expected, out, args.debug, args.make_expected_mean_match, args.median, mask)
