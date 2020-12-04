import sys
sys.path.insert(0,'../../..')
import pandas as pd
import re

import warnings
warnings.filterwarnings("ignore")

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pathlib import Path
import argparse
import pickle
from collections import defaultdict
from scipy.interpolate import interp1d

from plot_figs import *
from split_waves import *


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
    meta_files = []
    try:
        with open(args.inputs) as f:
            line = skip_comment(f)
            source_folder = line
            line = skip_comment(f)
            hemo_folder = line
            line = skip_comment(f)
            target_folder = line
            while(line):
                line = skip_comment(f)
                meta_files.append(line)               
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    return source_folder, hemo_folder, meta_files, target_folder

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

class NestedDict(defaultdict):
    def __init__(self):
        super().__init__(self.__class__)
    def __reduce__(self):
        return (type(self), (), None, None, iter(self.items()))

def save_figs(gcamp_mean, hemo_mean, gcamp_mean_raw, fit, target_folder, id, maximal_height_difference, min_slope, smoothing):
    id = os.path.join(target_folder, id)

    sws, start = scan_slow_wave_events(gaussian_filter(gcamp_mean, smoothing), maximal_height_difference = maximal_height_difference)
    
    sws_clean, sws_clean_normal = slow_waves_non_continuous(sws, start, gcamp_mean, "vector", min_slope, trim = False)[1:3]
    plot_sequence_of_clean_waves(hemo_mean, sws_clean, sws_clean_normal).savefig(id+"_clean_sequence.png")
    sws, sws_clean, _, _, _, _ = slow_waves_non_continuous(sws, start, gcamp_mean, "list", min_slope, trim = False)
    plot_sample_slow_waves(gcamp_mean_raw, sws, sws_clean).savefig(id+"_slow_wave_examples.png")

    plot_correlation_with_gcamp_per_wave(gcamp_mean, hemo_mean).savefig(id+"_correlations_with_hemo_per_wave.png")
    plot_fit(gcamp_mean, gcamp_mean_raw, fit).savefig(id+"_baseline_fit.png")

    plot_slow_waves_split(gcamp_mean, hemo_mean, smoothing, maximal_height_difference).savefig(id+"_slow_wave_split.png")
    plt.close('all')

def sharpen(vector, alpha = 1):
    details = vector - gaussian_filter(vector, 10)
    details[details<0] = 0
    vector += details*alpha
    return vector


def load_data(input_file, hemo_file, meta_file, dataset_path, debug):
    meta = pd.read_csv(meta_file, index_col = "run")
    if not os.path.isfile(dataset_path):
       print("Initialized datset")
       print(dataset_path)
       dataset = NestedDict()#defaultdict(lambda: defaultdict(dict))
    else:
       print("Loaded datset")
       with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
    
    #Using temporary files in debug mode
    if debug:
       temp_gcamp = os.path.join("temp_gcamp",os.path.basename(input_file))
       temp_hemo = os.path.join("temp_hemo",os.path.basename(hemo_file))
       Path("temp_hemo").mkdir(parents=True, exist_ok=True)
       Path("temp_gcamp").mkdir(parents=True, exist_ok=True)
   
    if debug and os.path.isfile(temp_gcamp) and os.path.isfile(temp_hemo):
         gcamp_mean = np.load(temp_gcamp)
         hemo_mean = np.load(temp_hemo)
    else:
        gcamp_mean = np.nanmean(np.load(input_file),axis=(1,2)) 
        print(".")
        hemo_mean = np.nanmean(np.load(hemo_file),axis=(1,2)) 
        print(".")
        np.save(temp_gcamp, gcamp_mean)
        np.save(temp_hemo, hemo_mean)
    return gcamp_mean, hemo_mean, meta, dataset
    

def process(input_file, hemo_file, meta_file, id, target_folder, init_dataset = False, maximal_height_difference = .05, min_slope = 10, smoothing = 4, figs = False, debug = False):

    dataset_path = os.path.join(target_folder,"dataset.pkl")
    gcamp_mean, hemo_mean, meta, dataset = load_data(input_file, hemo_file, meta_file, dataset_path, debug)
    print(input_file + "\n" + target_folder + "\n" + id + "\n")

    #Copy originals such that they can be plotted:
    first_fit = None
    gcamp_mean_raw = gcamp_mean.copy()

    #Sharpen gcamp signal (is percentage change hence smoothed), detrend hemodynamic signal
    gcamp_mean = sharpen(gcamp_mean, 10)
    hemo_mean = hemo_mean - gaussian_filter(hemo_mean, 250)

    #Remove baseline (several iterations via minfiltering)
    for i, (smooth, splines) in enumerate(zip([100,50,10,5,1],[25,30,35,40,70])):
        fit = fit_baseline_minfilter(gcamp_mean, smooth = smooth)[0]
        gcamp_mean = gcamp_mean - fit
        if type(first_fit) == type(None):
           first_fit = fit

    #Remove baseline (via rubberband fit)
    gcamp_mean = correct_baseline_rubberband(gcamp_mean, 10)[0]

    #Scan events, find corresponding hemodynamic patches, compute correlation
    sws_a, start = scan_slow_wave_events(gaussian_filter(gcamp_mean, smoothing), maximal_height_difference = maximal_height_difference)
    hemos_a, hemo_interp, corrs = get_correlations_continuous(start, sws_a, hemo_mean)

    print("n Waves")
    print(len(sws_a))

    sw_non_continuous = slow_waves_non_continuous(sws_a.copy(), start, gcamp_mean, "list", trim = True)
    sws, sws_c, sws_cn, sws_interp, lengths, height, nc_starts, nc_stops = sw_non_continuous#

    if figs:
       save_figs(gcamp_mean, hemo_mean, gcamp_mean_raw, first_fit, target_folder, id, maximal_height_difference, min_slope, smoothing)


    # Get start indices
    starts = []
    idx = start
    for s in sws_a:
        starts.append(idx)
        idx += len(s)
    stops = [start + width for start, width in zip(starts, lengths)]
    
    left_too_high = [(s[0] - s[-1]) > .75 for s in sws_cn]
    print(np.sum(left_too_high)/len(left_too_high))

    props = zip(sws_a, sws_c, sws_cn, sws_interp, hemos_a, hemo_interp, 
                height, lengths, starts, stops, corrs, nc_starts, nc_stops, left_too_high)
    for i, (sw_a, sw_c, sw_cn, sw_i, hemo_a, hemo_i, height, width, start, stop, corr, nc_start, nc_stop, l_too_high) in enumerate(props):
        full_id = id+"_sw_" + "{:04d}".format(i)
        dataset["sws"][full_id]["file_id"] = id
        dataset["sws"][full_id]["gcamp_mean"] = sw_c#trimmed, no normalization, no interpolation
        dataset["sws"][full_id]["gcamp_aligned"] = sw_a#not_trimmed aligned to gcamp aligned slow waves
        dataset["sws"][full_id]["shape"] = sw_cn#sws_clean_normal (non continuous normalized, trimmed)
        dataset["sws"][full_id]["gcamp_interpolated"] = sw_i# sws_interp (intepolated sws_clean_normal)
        dataset["sws"][full_id]["hemo_aligned"] = hemo_a#continuous
        dataset["sws"][full_id]["hemo_interpolated"] = hemo_i
        dataset["sws"][full_id]["height"] = height
        dataset["sws"][full_id]["width"] = width
        dataset["sws"][full_id]["start"] = start
        dataset["sws"][full_id]["stop"] = stop
        dataset["sws"][full_id]["correlation"] = corr
        dataset["sws"][full_id]["nc_start"] = nc_start
        dataset["sws"][full_id]["nc_stop"] = nc_stop
        dataset["sws"][full_id]["left_too_high"] = l_too_high
        
        run = int(re.match("exp_[0-9]*_run_([0-9]*)", id).groups()[0])
        exp = int(re.match("exp_([0-9]*)_run_[0-9]*", id).groups()[0])
        dataset["sws"][full_id]["iso"] = meta["iso"][run]

    dataset["file"][id]["sws_continuous"]["start"] = start
    dataset["file"][id]["sws_continuous"]["sws"] = sws_a#aligned continuous
    dataset["file"][id]["hemo_mean"] = hemo_mean
    dataset["file"][id]["gcamp_mean"] = gcamp_mean

    dataset["exp"][int(exp)][run]["sws_continuous"]["start"] = start
    dataset["exp"][int(exp)][run]["sws_continuous"]["sws"] = sws_a#aligned continuous
    dataset["exp"][int(exp)][run]["hemo_mean"] = hemo_mean
    dataset["exp"][int(exp)][run]["gcamp_mean"] = gcamp_mean
        
    with open(dataset_path, "wb") as f:
       pickle.dump(dataset, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--figs', action="store_true")
    parser.add_argument('--init_dataset', action="store_true")
    args = parser.parse_args()
    use_default_masks = False

    source_folder, hemo_folder, meta, target_folder = read_configfile(args)
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    
    input_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder)]
    input_files.sort()
    hemo_files = [os.path.join(hemo_folder, os.path.basename(f)) for f in input_files]
    hemo_files.sort()

    meta_files = []
    for f in input_files:
        for m in meta:
            if m == "":
               continue
            if os.path.basename(m).split(".")[0] in f:
               meta_files.append(m)
    print(meta_files)
     

    for input_file, hemo_file, meta_file in zip(input_files, hemo_files, meta_files):
        if not os.path.isfile(input_file) or not os.path.isfile(hemo_file):
            print("Missing file for ")
            print(os.path.basename(input_file))  
        print(".", end="")
        process(input_file, hemo_file, meta_file, os.path.basename(input_file).split(".")[0], target_folder, args.init_dataset, figs = args.figs, debug = args.debug)
