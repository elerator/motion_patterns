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



def get_filenames(args):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    input_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    input_paths = [os.path.join(source_folder, f) for f in input_files]
    output_paths = [os.path.join(target_folder, f) for f in input_files]
    return source_folder, target_folder, input_paths, output_paths

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

def process(input_file, output_file, debug):
    print(input_file)
    print(output_file)
    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line is the input directory.'
                                        +'The second line is the output directory. All .npy files in the input directory are being processed.')

    parser.add_argument('--debug',  action="store_true")
    parser.add_argument('-parallel_processes', nargs="?", const=5, type=int)
    args = parser.parse_args()
    
    try:
        with open(args.inputs) as f:
            line = f.readline().rstrip('\n')
            source_folder = line
            print("Using source_folder " + source_folder)
            line = f.readline().rstrip('\n')
            target_folder = line
            print("Using target_folder " + target_folder)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    
    files = []
    outfiles = []
    for f in os.listdir(source_folder):
        if f.endswith(".npy") and not "_mean" in f:
             outpath  = os.path.join(target_folder, f)
             if os.path.isfile(outpath):
                print(f + " exists already", end = " ")
                print("Delete file if you would like to recompute\n")
                continue
             print("Output set to "+ outpath)
             outfiles.append(outpath)
             files.append(os.path.join(source_folder, f))
    
    processes = []
    max_processes = args.parallel_processes
    n_processes = 0
    for inpath, outpath in zip(files, outfiles):
        if args.parallel_processes > 1:
           if n_processes > max_processes:
                 for job in processes:
                     job.join()#wait for all to finish
                 n_processes = 0
           p1 = Process(target=lambda: process(inpath, outpath, args.debug))
           p1.start()
           processes.append(p1)
           n_processes += 1
        else:
           print("Single process")
           process(inpath, outpath, args.debug)
