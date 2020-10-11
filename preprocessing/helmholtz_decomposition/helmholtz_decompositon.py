import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process


#imports
from pathlib import Path
import argparse

def process(inpath, outroot,mask_path = None,  debug = False):
    print("......")
    outfolders = [os.path.join(outroot, name) for name in ["solenoidal_gradient", "solenoidal_function", "curl_free_gradient", "curl_free_function"]]
    
    for target_folder in outfolders:
          Path(target_folder).mkdir(parents=True, exist_ok=True)

    vfields = np.load(inpath)
    if mask_path:
        mask = np.array(Image.open(mask_path)) == 255
	
    print("Processing " + str(vfields.shape[1]) + " vector fields")
    if debug:
        vfields = vfields[:,:100,:,:]
    solenoidal_gradient, solenoidal_function, curl_free_gradient, curl_free_function = helmholtz_decomposition(vfields)
    if mask_path:
        solenoidal_function[:, mask] = np.nan
        curl_free_function[:, mask] = np.nan
    np.save(os.path.join(outfolders[0], os.path.basename(inpath)), solenoidal_gradient)
    np.save(os.path.join(outfolders[1], os.path.basename(inpath)), solenoidal_function)
    np.save(os.path.join(outfolders[2], os.path.basename(inpath)), curl_free_gradient)
    np.save(os.path.join(outfolders[3], os.path.basename(inpath)), curl_free_function)
    print("Teminated successfully")


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
            mask_folder = line
            print("Using mask_folder " + mask_folder)
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

    files = [] 
    masks = []
    print(source_folder)
    print(os.listdir(source_folder)) 
    for f in os.listdir(source_folder):
        if f.endswith(".npy") and not "_mean" in f:
             if os.path.isfile(os.path.join(os.path.join(os.path.dirname(f),"solenoidal_gradient"), os.path.basename(f))):
                print("Outputs for file " + f + " exist already")
                print("Delete files if you would like to recompute\n")
                continue
             print("Found input tensor "+f)
             files.append(os.path.join(source_folder, f))
             
             maskname = f.split(".")[0] + "_mask.png"
             if not os.path.isfile(os.path.join(mask_folder, maskname)):
                print("mask not found")
                sys.exit()
             masks.append(os.path.join(mask_folder, maskname))
    processes = []
 
    for inpath, mask_path in zip(files, masks):
        print("------------------")
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, target_folder,mask_path, args.debug))
           p1.start()
           processes.append(p1)
        else:
           print("...")
           process(inpath,target_folder, mask_path, args.debug)
