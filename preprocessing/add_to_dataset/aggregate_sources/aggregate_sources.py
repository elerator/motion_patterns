import sys
sys.path.insert(0,'../../..')

import gc

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pathlib import Path
import argparse

import pickle
from threading import Thread

from PIL import Image

def get_filenames(inputs):
    try:
        with open(inputs) as f:
            line = f.readline().rstrip('\n')
            source_folder = line
            print("Using source_folder " + source_folder)
            line = f.readline().rstrip('\n')
            target_folder = line
            print("Using target_folder " + target_folder)
            line = f.readline().rstrip('\n')
            dataset = line
            print("Using dataset " + dataset)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    return source_folder, target_folder, dataset

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

def interpolate(y, n = 10):
    return interp1d(np.arange(len(y)), y)(np.linspace(0,len(y)-1, n))


def dump_secure(data, path, mode = "wb"):
    """ Pickle dump without interrupt. Prevents corrupt files due to KeyBoardInterrupt
    Args:
         data: Data to be dumped via pickle
         path: Filepath
         mode: Mode for file writing
    """
    def _dump_secure():
        nonlocal data
        nonlocal path
        nonlocal mode
        with open(path, "wb") as f:
           pickle.dump(data,f)
    a = Thread(target = _dump_secure)
    a.start()
    a.join()
     
def devide_by_max(t):
    t = t / np.nanmax(t)
    return t
 
def do_normalize_frames(sources, sinks):
      sources = np.array([devide_by_max(s) for s in sources])
      sinks = np.array([devide_by_max(s) for s in sinks])
      #sources = sources > .1
      #sinks = sinks > .1
      sources = normalize_nan(np.nanmean(sources, axis = 0))
      sinks = normalize_nan(np.nanmean(sinks, axis = 0))
 
      return sources, sinks


def process(iteration, dataset, input_file, target_folder, output_file, dataset_path, epsilon = 0, per_hemisphere = False, original_size = False, normalize_frames = False, figs = False, debug = False):
   try:
       sources_sinks = np.load(input_file)
   except:
       print("Retry loading tensors")
       time.sleep(random.uniform(0,1))


   sws_id = os.path.basename(input_file).split(".")[0]

   target_folder_sources = os.path.join(target_folder, "sources")
   target_file_sources = os.path.join(target_folder_sources, output_file)
   target_folder_sinks = os.path.join(target_folder, "sinks")
   target_file_sinks = os.path.join(target_folder_sinks, output_file)
   Path(target_folder_sources).mkdir(parents=True, exist_ok = True)
   Path(target_folder_sinks).mkdir(parents=True, exist_ok = True) 


   sources = sources_sinks.copy()
   sources[sources > 0 + epsilon] = 0#Only negative values
   sources *= -1
   sinks = sources_sinks
   sinks[sinks < 0] = 0
   
   if normalize_frames:
      sources, sinks = do_normalize_frames(sources, sinks)
   else:
      sources = np.nanmean(sources, axis = 0)
      sinks = np.nanmean(sinks, axis = 0)
      if not per_hemisphere:
         sources = normalize_nan(np.nanmean(sources, axis = 0))
         sinks = normalize_nan(np.nanmean(sinks, axis = 0)) 
      else:
         sources[:,:sources.shape[1]//2] = normalize_nan(sources[:,:sources.shape[1]//2])
         sources[:,sources.shape[1]//2:] = normalize_nan(sources[:,sources.shape[1]//2:])
         sinks[:,:sinks.shape[1]//2] = normalize_nan(sinks[:,:sinks.shape[1]//2])
         sinks[:,sinks.shape[1]//2:] = normalize_nan(sinks[:,sinks.shape[1]//2:])

   if figs:
      sources[np.isnan(sources)] = 0
      sinks[np.isnan(sinks)] = 0
      sources *= 255
      sinks *= 255
      sources = Image.fromarray(sources.astype(np.uint8))
      sinks = Image.fromarray(sinks.astype(np.uint8))
      if not original_size:
         sources = sources.resize((64,64))
         sinks = sinks.resize((64,64))
      sinks.save(target_folder_sinks+ ".png")
      sources.save(target_file_sources+ ".png")
    
   #output_files.append(output_file)
   #data_sources.append(sources)
   #data_sinks.append(sinks)
   print(".", end="")

   dataset["sws"][sws_id]["mean_sources"] = np.array(sources)/255
   dataset["sws"][sws_id]["mean_sinks"] = np.array(sinks)/255
   return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line is the input directory.'
                                        +'The second line is the output directory. All .npy files in the input directory are being processed.')

    parser.add_argument('--debug',  action="store_true")
    parser.add_argument('--normalize_frames',  action="store_true")
    parser.add_argument('--original_size',  action="store_true")
    parser.add_argument('--per_hemisphere',  action="store_true")
    parser.add_argument('--figs',  action="store_true")
    args = parser.parse_args()
    
    source_folder, target_folder, dataset_path = get_filenames(args.inputs)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    try:
       with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
    except Exception as e:
       print("Fatal error: Problem loading dataset")
       print(str(e))
       sys.exit(0)
    
    files = []
    outfiles = []
    found_files = os.listdir(source_folder)
    found_files.sort()
    for f in found_files:
        if f.endswith(".npy"):
             outpath = os.path.join(os.path.join(target_folder, "per_frame"), f.split(".")[0] + ".png")
             if os.path.isfile(outpath):
                print(f + " exists already", end = " ")
                print("Delete file if you would like to recompute\n")
                continue
             outfiles.append(f.split(".")[0])
             files.append(os.path.join(source_folder, f))
    

    for i, (inpath, outfile) in enumerate(zip(files, outfiles)):
        print(str((i/len(files)*100))+"%")
        dataset = process(i, dataset, inpath, target_folder, outfile, dataset, 0, args.per_hemisphere, args.original_size, args.normalize_frames, args.figs, args.debug)

    dump_secure(dataset, dataset_path)
    print("Saved dataset successfully")

