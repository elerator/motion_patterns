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
import random

def process(inpath, outroot,mask_path = None, per_hemisphere = True,  debug = False):
    outfolders = [os.path.join(outroot, name) for name in ["solenoidal_gradient", "solenoidal_function", "curl_free_gradient", "curl_free_function"]]
    
    for target_folder in outfolders:
          Path(target_folder).mkdir(parents=True, exist_ok=True)

    vfields = None
    for _ in range(1):
      try:
        vfields = np.load(inpath)
        if mask_path:
           if not os.path.isfile(mask_path):
              print("No mask found")
              sys.exit()
           im = Image.open(mask_path)
           mask = np.array(im) == 255
           im.close()
      except:
        print("Retry loading")
        #time.sleep(random.uniform(0,1))
    if type(vfields) == type(None):
        raise Exception("Could not load vfields")	
    if debug:
        vfields = vfields[:,:10,:,:]
    if per_hemisphere:
        sgl, sfl, cgl, cfl = helmholtz_decomposition(vfields[:,:,:,:vfields.shape[3]//2])
        sgr, sfr, cgr, cfr = helmholtz_decomposition(vfields[:,:,:,vfields.shape[3]//2:])
        solenoidal_function = np.concatenate([sfl, sfr], axis=2)
        curl_free_function = np.concatenate([cfl, cfr], axis=2)
        curl_free_gradient = np.concatenate([cgl, cgr], axis=3)
        solenoidal_gradient = np.concatenate([sgl, sgr], axis=3)
    else:
        solenoidal_gradient, solenoidal_function, curl_free_gradient, curl_free_function = helmholtz_decomposition(vfields)
    if mask_path:
        solenoidal_function[:, mask] = np.nan
        curl_free_function[:, mask] = np.nan
        curl_free_gradient[:,:,mask] = np.nan
        solenoidal_gradient[:,:,mask] = np.nan
    np.save(os.path.join(outfolders[0], os.path.basename(inpath)), solenoidal_gradient)
    np.save(os.path.join(outfolders[1], os.path.basename(inpath)), solenoidal_function)
    np.save(os.path.join(outfolders[2], os.path.basename(inpath)), curl_free_gradient)
    np.save(os.path.join(outfolders[3], os.path.basename(inpath)), curl_free_function)
    del vfields
    print(".", end="")

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
            mask_path = line
            print("Using mask " + mask_path)
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
    print(source_folder)
    for f in os.listdir(source_folder):
        if f.endswith(".npy") and not "_mean" in f:
             if os.path.isfile(os.path.join(os.path.join(target_folder,"solenoidal_gradient"), os.path.basename(f))):
                print("Outputs for file " + f + " exist already")
                print("Delete files if you would like to recompute")
                continue
             files.append(os.path.join(source_folder, f))
             
             #maskname = f.split(".")[0] + "_mask.png"
             #if not os.path.isfile(os.path.join(mask_folder, maskname)):
             #   print("mask not found")
             #   sys.exit()
             #masks.append(os.path.join(mask_folder, maskname)
    print("Found " + str(len(files)) + " files to process")
    processes = []
    n_processes = 0
    max_processes = 100 
    for i, inpath in enumerate(files):
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, target_folder,mask_path, per_hemisphere =  True, debug = args.debug))
           p1.deamon = True
           p1.start()
           processes.append(p1)
           n_processes += 1
           if n_processes > max_processes:
                 for job in processes:
                     job.join()#wait for all to finish
                 n_processes = 0
                 processes = []
                 print("")
                 print(str(i/len(files)*100)+"%")
        else:
           process(inpath,target_folder, mask_path, per_hemisphere = True, debug = args.debug)
