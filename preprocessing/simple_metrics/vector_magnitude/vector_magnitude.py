import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process


#imports
from pathlib import Path
import argparse

def process(inpath, outpath, debug = False):
    flow = np.load(inpath)
    out = np.sqrt(flow[0]**2+flow[1]**2)
    np.save(outpath, out)
    print("saved " + outpath + " successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line is the input directory.'
                                        +'The second line is the output directory. All .npy files in the input directory are being processed.')

    parser.add_argument('--debug',  action="store_true")
    parser.add_argument('--parallel_processes',  action="store_true")
    args = parser.parse_args()

    try:
        with open(args.inputs) as f:
            line = f.readline().rstrip('\n')
            source_folder = line
            print("Using source_folder " + source_folder)
            line = f.readline().rstrip('\n')
            target_folder = line
            print("Using target_folder " + target_folder)
            line = f.readline().rstrip('\n')
            
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
    #masks = [np.array(Image.open(os.path.join(mask_folder, os.path.basename(f).split(".")[0]+"_mask.png"))) == 255 for f in files]
    processes = []
    for inpath, outpath in zip(files, outfiles):
        print("------------------")
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, outpath, args.debug))
           p1.start()
           processes.append(p1)
        else:
           process(inpath, outpath, args.debug)
