import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process

#imports
from pathlib import Path
import argparse
import time

def process(inpath, outpath, per_hemisphere = True, debug = False):
    tensor = np.load(inpath,mmap_mode="c")

    print("Processing tensor with " + str(len(tensor)) + " frames")
    time.sleep(1)
    if debug:
       tensor = tensor[:100]
    if per_hemisphere:
        x_comp_left, y_comp_left = horn_schunck(tensor[:,:(tensor.shape[1]//2),:])
        x_comp_right, y_comp_right = horn_schunck(tensor[:,(tensor.shape[1]//2):,:])
        x_comp = np.concatenate([x_comp_left,x_comp_right], axis=1)
        y_comp = np.concatenate([y_comp_left,y_comp_right], axis=1)
    else:
        x_comp, y_comp = horn_schunck(tensor)
    out = np.array([y_comp, x_comp])
    np.save(outpath, out)
    del tensor
    del x_comp
    del y_comp


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
        if args.parallel_processes and args.parallel_processes > 1:
           if n_processes > max_processes:
                 for job in processes:
                     job.join()#wait for all to finish
                 processes = [] 
                 n_processes = 0
           p1 = Process(target=lambda: process(inpath, outpath, per_hemisphere = True, debug = args.debug))
           p1.start()
           processes.append(p1)
           n_processes += 1
        else:
           print("Single process")
           process(inpath, outpath, per_hemisphere = True, debug = args.debug)
