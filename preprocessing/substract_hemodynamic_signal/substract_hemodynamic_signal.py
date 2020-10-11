import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *

#imports
from multiprocessing import Process
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter

def process(inpath, hemo, outpath, corr_out, predict_with_mean_hemo = False, debug = False):
    """ Performs preprocessing. Substracts prediction of a linear regression model based on the hemodynamic signal from the GCamp signal. 
    Args:
     predict_with_mean_hemo: If True predictions are made using the framewise mean of the hemodynamic variable to predict each pixel vector using a seperate linear regression model. Otherwise the pixel-vector at the same position in the hemodynamic signal is used to do so.
    inpath: input path
    outpath: output path
    debug: Debug mode
    """
    print(inpath)
    gcamp = np.load(inpath)
    hemo = np.load(hemo)
    if predict_with_mean_hemo:
       hemo_mean = np.mean(hemo, axis = (1,2))
       hemo_mean = hemo_mean-gaussian_filter(hemo_mean,1000)#Remove potential slow trend
    rs = np.ndarray(hemo[0].shape)
    rs.fill(np.nan)
    for y in range(gcamp.shape[1]):
        print(np.round(100*(y/gcamp.shape[1])))
        for x in range(gcamp.shape[2]):
            if predict_with_mean_hemo:
               hemo_vector = hemo_mean
            else:
               hemo_vector = hemo[:, y, x]
            #hemo_vector = hemo_vector - gaussian_filter(hemo_vector, 1000)
            if np.any(np.isnan(hemo_vector)) or np.any(np.isnan(gcamp[:,y,x])):
               continue
            slope, intercept, rval, pval, std = linregress(hemo_vector, gcamp[:,y,x])
            rs[y, x] = rval
            prediction = intercept + slope * hemo_vector
            gcamp[:, y,x] = gcamp[:, y, x] - prediction
    np.save(corr_out, rs)
    rs[np.isnan(rs)] = 0
    rs *= 255
    Image.fromarray(rs.astype(np.uint8)).save(corr_out+ ".png")
    np.save(outpath, gcamp)

def skip_comment(file):
    while True:
       line = file.readline().rstrip('\n')
       if not "#" in line:
          break
       if not line:
          raise Exception("No line to read found") 
    return line
    
def read_configfile(args):
    source_folder = ""
    hemo_folder = ""
    target_folder = ""
    try:
        with open(args.inputs) as f:
            line = skip_comment(f)
            source_folder = line
            line = skip_comment(f)
            hemo_folder = line
            line = skip_comment(f)
            target_folder = line
            line = skip_comment(f)
            corr_folder = line
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    return source_folder, hemo_folder, target_folder, corr_folder

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
    parser.add_argument('-inputs', help='Path to config file.')
    parser.add_argument('--debug',  action="store_true")
    parser.add_argument('--use_mean_hemo',  action="store_true")
    parser.add_argument('--override',  action="store_true")
    parser.add_argument('--parallel_processes',  action="store_true")
    args = parser.parse_args()

    source_folder, hemo_folder, target_folder, corr_folder = read_configfile(args) 
    print("Using the following input and target folders")
    print(source_folder)
    print(hemo_folder)
    print(target_folder)
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    Path(corr_folder).mkdir(parents=True, exist_ok=True)
    source_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    target_files = [os.path.join(target_folder, f) for f in source_files]
    corr_files = [os.path.join(corr_folder, f) for f in source_files]
    hemo_files = [os.path.join(hemo_folder, f) for f in source_files]
    source_files = [os.path.join(source_folder, f) for f in source_files]

    if np.any([not os.path.isfile(f) for f in hemo_files]):
        print("There is a hemodynamic file missing")
        sys.exit()
    if not args.override:
      target_files = remove_existing_files(target_files)
    
    processes = []
    for inpath, hemo, outpath, corr_path in zip(source_files, hemo_files, target_files, corr_files):
        if not outpath:
            print("Skipping " + inpath + " because outputfiles exist already")
            continue
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, hemo, outpath, corr_path, args.use_mean_hemo,  args.debug))
           p1.start()
           processes.append(p1)
        else:
           process(inpath, hemo, outpath, corr_path, args.use_mean_hemo, args.debug)
