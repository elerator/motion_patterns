import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *

#imports
from multiprocessing import Process
from pathlib import Path
import argparse

from scipy.interpolate import interp1d
from scipy.signal import detrend
import numpy as np
import h5py


def resize_by_interpolation(tensor, target_lengths=30000,kind="cubic"):
    """ Resizes 3d tensor along first dimension using the specified kind of interpolation. Intended to increase tensor size.
    Args:    
        tensor: Input 3d tensor (numpy ndarray)
        target_lengths: Desired size of output tensor along first axis
        kind: kind of interpolation "cubic" or "linear"
    Returns:
        resized tensor
    """
    output_tensor = np.ndarray((target_lengths, tensor.shape[1], tensor.shape[2]))
    x_vals = np.arange(len(tensor))
    interlacing_factor = target_lengths//len(tensor)
    n_elem_start = interlacing_factor//2
    n_elem_end = interlacing_factor - n_elem_start

    for y in range(tensor.shape[1]):
        print(".", end="")
        for x in range(tensor.shape[2]):
            y_vals = tensor[:,y,x]
            interpolated = np.ndarray(target_lengths)
            interpolated[:n_elem_start] = [y_vals[0] for x in range(n_elem_start)]#First entries cannot be interpolated
            interpolated[-n_elem_end:] = [y_vals[-1] for x in range(n_elem_end)]
            f = interp1d(x_vals, y_vals, kind=kind)
            interpolated[n_elem_start:-n_elem_end] = f(np.linspace(0,len(tensor)-1,(len(tensor)-1)*interlacing_factor))
            output_tensor[:,y,x] = interpolated
    return output_tensor

def process(inpath, gcamp_out, hemo_out, debug = False):
    f = h5py.File(inpath,'r')
    f.keys()
    f.get('gcamp')
    gcamp = np.array(f.get('gcamp'))
    gcamp = np.rot90(gcamp,-1,axes=(1,2))

    np.save(gcamp_out, gcamp)
    gcamp = None

    hemo = np.array(f.get('hemo'))
    hemo = np.rot90(hemo,-1,axes=(1,2))

    hemo = resize_by_interpolation(hemo)
    hemo = detrend(hemo, axis=0)
    np.save(hemo_out, hemo)

def skip_comment(file):
    while True:
       line = file.readline().rstrip('\n')
       if not "#" in line:
          break
       if not line:
          raise Exception("No line to read found") 
    return line
    
def read_configfile(args):
    target_gcamp = ""
    target_hemo = ""
    source_paths = []
    target_names = []
    try:
        with open(args.inputs) as f:
            line = skip_comment(f)
            target_gcamp = line
            line = skip_comment(f)
            target_hemo = line
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.rstrip('\n')
                if "#" in line:
                    continue
                source, target = line.split(" ")
                source_paths.append(source)
                target_names.append(target)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    return target_gcamp, target_hemo, source_paths, target_names

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
    parser.add_argument('--override',  action="store_true")
    parser.add_argument('--parallel_processes',  action="store_true")
    args = parser.parse_args()

    target_gcamp, target_hemo, source_paths, target_names = read_configfile(args)
    print("Target folders:")
    print(target_gcamp)
    print(target_hemo)
    print("Files")
    
    for f in target_names:
        print(f) 
    
    Path(target_gcamp).mkdir(parents=True, exist_ok=True)
    Path(target_hemo).mkdir(parents=True, exist_ok=True)
   
    target_gcamp = [os.path.join(target_gcamp,f) for f in target_names]
    target_hemo = [os.path.join(target_hemo, f) for f in target_names]
    
    if not args.override:
      target_gcamp = remove_existing_files(target_gcamp)
      target_hemo = remove_existing_files(target_hemo) 
    processes = []
    for inpath, gcamp, hemo in zip(source_paths, target_gcamp, target_hemo):
        if not gcamp or not hemo:
            print("Skipping " + inpath + " because outputfiles exist already")
            continue
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, gcamp, hemo, args.debug))
           p1.start()
           processes.append(p1)
        else:
           process(inpath, gcamp, hemo, args.debug)
