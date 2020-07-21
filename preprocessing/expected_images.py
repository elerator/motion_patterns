import sys
sys.path.append('..')

#import of utilities
from utils.data_transformations import *
import utils.data_transformations
from utils.diverse import *
import utils.diverse

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


from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

def value_range_of_frame_means(filepaths, maskpath=False):
    """ Retrieve value-range of several files. This method is slow but it opens one file at a time only such that the memory requirements are limited.
    Args:
        filepaths: List of filepaths to tif files
    Returns:
        min: Minimum value
        max: Maximal value
    """
    prelim_min = float("inf")
    prelim_max = -float("inf")
    for filepath in filepaths:
        print(".", end="")
        sys.stdout.flush()
        frames = np.array(skimage.io.imread(os.path.join(filepath)), dtype=np.double)
        print(".", end= "")
        sys.stdout.flush()
        mean = np.mean(frames,axis=0)#pixelwise mean
        print(".", end = "")
        sys.stdout.flush()
        frames = framewise_difference(frames, mean, bigdata=True)
        mean = None

        if maskpath:
            mask = np.array(Image.open(maskpath))[:,:,0] == 255
            frames = apply_mask(frames, mask, nan=True)

        min_val = np.min(np.nanmean(frames,axis=(1,2)))#maximal value of framewise mean
        max_val = np.max(np.nanmean(frames,axis=(1,2)))

        if min_val < prelim_min:
            prelim_min = min_val
        if max_val > prelim_max:
            prelim_max = max_val
    return prelim_min, prelim_max

def expected_images(filepaths, min_val, max_val,bins=100, maskpath = False):
    """ Retrieve expected images for a given brighness value.
    Args:
        filepaths: List of filepaths
        min_val: Minimum value of frame means
        max_val: Maximum value of frame means
        bins: Number of bins between min_val and max_val for which the expected image is calculated
    """
    n_per_bin = np.zeros(shape = [bins])
    bin_upper_boundaries = np.linspace(0, bins,bins+1)
    output_tensor = None
    for filepath in filepaths:
        print(".", end ="")
        sys.stdout.flush()
        frames = np.array(skimage.io.imread(os.path.join(filepath)), dtype= np.double)
        print(".", end="")
        sys.stdout.flush()
        mean = np.mean(frames,axis=0)#pixelwise mean
        print(".", end="")
        sys.stdout.flush()
        frames = framewise_difference(frames, mean, bigdata=True)
        mean = None
        print(".", end="")
        sys.stdout.flush()

        if maskpath:
            mask = np.array(Image.open(maskpath))[:,:,0] == 255
            frames = apply_mask(frames, mask, nan=True)

        if type(output_tensor) == type(None):
            output_tensor = np.zeros(shape = [bins,frames.shape[1],frames.shape[2]], dtype=np.double)
        for i, frame in enumerate(frames):
            if (i % 500) == 0:
                print("*",end="")
                sys.stdout.flush()
            frame_mean = np.nanmean(frame)
            assert frame_mean <= max_val
            assert frame_mean >= min_val

            frame_mean -= min_val
            frame_mean /= (max_val-min_val)
            frame_mean *= bins
            frame_mean = int(frame_mean)
            if frame_mean >= bins:
                print(frame_mean)
                continue

            n_per_bin[frame_mean] += 1
            output_tensor[frame_mean] += frame

    output_tensor = output_tensor/n_per_bin[:, np.newaxis, np.newaxis] #TODO

    return output_tensor, bin_upper_boundaries, n_per_bin

def interpolate_tensor(tensor, size, axis=0, smoothing=None):
    """ Resizes and intepolates along axis
    Args:
        tensor: 3d tensor
        size: Desired output size along axis
        axis: Axis along which the tensor is resized
        smoothing: Sigma of the gaussian used for smoothing before resizing
    """
    if smoothing:
        if axis == 0:
            tensor = gaussian_filter(tensor, smoothing, [1,0,0], mode="mirror")
        elif axis == 1:
            tensor = gaussian_filter(tensor, smoothing, [0,1,0], mode="mirror")
        elif axis == 2:
            tensor = gaussian_filter(tensor, smoothing, [0,0,1], mode="mirror")
    x = np.linspace(0, tensor.shape[axis], tensor.shape[axis])
    x_new = np.linspace(0, tensor.shape[axis], size)
    out = interp1d(x, tensor, axis=axis)(x_new)
    return out


def pixelwise_polynomial(tensor, polynomial_degree, ignore_larger = 7):
    output = np.ndarray(tensor.shape)
    len_output_vectors = output.shape[0]
    tensor = tensor[:ignore_larger,:,:].copy()

    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            vector = tensor[:,y,x]
            if np.isnan(vector[0]):
                output[:,y,x] = vector
            else:
                #plt.plot(np.arange(len(vector)),vector)
                parameters = np.polyfit(np.arange(len(vector)), vector, polynomial_degree)
                output[:,y,x] = np.polyval(parameters, np.arange(len_output_vectors))
    return output

def save_expected_images(files, source_folder, output_folder, polyfit = False, mask = None, min_val=None, max_val=None):
    filenames_raw = files.copy()
    files = [os.path.join(source_folder, f) for f in files]
    if not min_val or not max_val:
        min_val, max_val = value_range_of_frame_means(files, maskpath=mask)
    print("min_val "+str(min_val), end = " ")
    print("max_val "+str(max_val))
    sys.stdout.flush()

    output_tensor = None
    if polyfit:
        output_tensor, upper_bin_boundaries, n_per_bin = expected_images(files, min_val,max_val,bins=10, maskpath=mask)
        output_tensor = pixelwise_polynomial(output_tensor, 2, 6)
        output_tensor = interpolate_tensor(output_tensor, 1000)
    else:
        output_tensor, upper_bin_boundaries_large, n_per_bin_large = expected_images(files, min_val,max_val,bins=25, maskpath=mask)
        output_tensor = interpolate_tensor(output_tensor, 1000)

    for file in filenames_raw:
        outfilename = os.path.join(output_folder,os.path.splitext(file)[0]+"_expected.npy")
        print("\nSaving " + outfilename)
        np.save(outfilename,output_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        + 'If flag --mask is set line three contains the path to the mask png'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')
    parser.add_argument('--polyfit', action="store_true")
    parser.add_argument('--mask', action="store_true")
    parser.add_argument('-min_val', nargs='?', default=None, type=float)
    parser.add_argument('-max_val', nargs='?', default=None, type=float)


    args = parser.parse_args()
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
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
                files_for_expected_img = line.split(" ")
                files.append(files_for_expected_img)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()

    for files_for_expected_img in files:
        for name in files_for_expected_img:
            print(name)
            if not os.path.isfile(os.path.join(source_folder,name)):
                print(name +" is not a file")
                sys.exit()

    Path(target_folder).mkdir(parents=True, exist_ok=True)

    #Write some metainfo
    metainfo = {"source_folder": source_folder}
    info = json.dumps({**metainfo,**vars(args)}, indent=4)
    with open(os.path.join(target_folder,"metainfo.txt"), "w") as f:
        f.write(info)

    for files_for_expected_img in files:
        save_expected_images(files_for_expected_img, source_folder, target_folder, polyfit = args.polyfit, mask=mask, min_val=args.min_val, max_val=args.max_val)
