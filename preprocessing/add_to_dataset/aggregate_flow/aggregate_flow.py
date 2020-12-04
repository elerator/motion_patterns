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

import time
import random

from scipy.interpolate import interp1d
import pickle
from threading import Thread


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


def devide_by_max(v):
   v = np.array(v)
   v = v/np.max(v)
   return v

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
      

def process(input_file, target_folder, output_file, dataset_path, epsilon = 0, figs = False, debug = False):
   try:
       y_comp, x_comp = np.load(input_file)
   except:
       print("Retry loading tensors")
       time.sleep(random.uniform(0,1))

   sws_id = os.path.basename(input_file).split(".")[0]
   try:
      with open(dataset_path, "rb") as f:
           dataset = pickle.load(f)

   except Exception as e:
      print("Fatal error: Problem loading dataset")
      print(str(e))
      sys.exit(0)

   print(".", end="")

   left, right, up, down = aggregate_components(y_comp, x_comp, epsilon, left_hemisphere = True)#Mean per pixel vector components per frame
   left_ip, right_ip, up_ip, down_ip = [interpolate(v, 10) for v in [left, right, up, down]]#Interpolated

   n_up_ip, n_down_ip, n_left_ip, n_right_ip = devide_by_max([up_ip, down_ip, left_ip, right_ip])#for figs
   n_up, n_down, n_left, n_right = devide_by_max([up, down, left, right])


   dataset["sws"][sws_id]["flow_components"]["per_frame"]["left_hemisphere"] = np.array([left, right, up, down])
   dataset["sws"][sws_id]["flow_components"]["per_frame_interpolated"]["left_hemisphere"] = np.array([left_ip, right_ip, up_ip, down_ip])

   mean_left, mean_right, mean_up, mean_down = [np.mean(t) for t in [left, right, up, down]]#Mean per pixel component per wave
   #save plots
   if figs:
      out_components_per_frame = os.path.join(target_folder, "components_per_frame")
      Path(out_components_per_frame).mkdir(parents=True, exist_ok=True)
      out_components_per_frame = os.path.join(out_components_per_frame, output_file + ".png")
      imgs = [render_arrow_components(up, down, left, right) for up, down, left, right in zip(n_up, n_down, n_left, n_right)]
      Image.fromarray(np.hstack(imgs)).save(out_components_per_frame)

      out_components_per_frame = os.path.join(target_folder, "components_per_frame_interpolated")
      Path(out_components_per_frame).mkdir(parents=True, exist_ok=True)
      out_components_per_frame = os.path.join(out_components_per_frame, output_file + ".png")
      imgs = [render_arrow_components(up, down, left, right) for up, down, left, right in zip(n_up_ip, n_down_ip, n_left_ip, n_right_ip)]
      Image.fromarray(np.hstack(imgs)).save(out_components_per_frame)

      out_components_per_wave = os.path.join(target_folder, "components_per_wave")
      Path(out_components_per_wave).mkdir(parents=True, exist_ok=True)
      out_components_per_wave = os.path.join(out_components_per_wave, output_file + ".png")
      vals = [mean_up, mean_down, mean_left, mean_right]
      vals /= np.max(vals)
      imgs = render_arrow_components(*vals)
      Image.fromarray(imgs).save(out_components_per_wave)
  
   dataset["sws"][sws_id]["flow_components"]["per_wave"]["left_hemisphere"] = np.array([mean_left, mean_right, mean_up, mean_down])

   dump_secure(dataset, dataset_path)

def aggregate_components(y_comp, x_comp, epsilon = 0, left_hemisphere = True, poly = True):
   """ Aggregates positive and negative vector components for each frame or tensor"""
   if left_hemisphere:
      #vfields[n,y_comp,y,x] 
      y_comp = y_comp[:,:y_comp.shape[1]//2]
      x_comp = x_comp[:,:x_comp.shape[1]//2]
   if poly:
      x_comp = x_comp ** 3
      y_comp = y_comp ** 3

   mask = np.isnan(y_comp[0])
   sum_pixels = np.sum(~np.isnan(mask))

   down = y_comp.copy()
   down[down < 0 + epsilon] = None#Select only positive vals
   down = np.nansum(down, axis = (1,2))/sum_pixels

   up = y_comp.copy()
   up[up > 0 - epsilon] = None#Select only negative vals
   up = np.nansum(up, axis =(1,2))/sum_pixels

   left = x_comp.copy()
   left[left > 0 - epsilon] = None#select only negative vals
   left = np.nansum(left, axis = (1,2))/sum_pixels

   right = x_comp.copy()
   right[right < 0 + epsilon] = None#select only positive vals
   right = np.nansum(right, axis = (1,2))/sum_pixels
   
   return np.abs(left), np.abs(right), np.abs(up), np.abs(down)


def test_aggregate_flow():
    """ Test aggregate flow"""
    y_comp, x_comp = np.array([np.ones((2,2))*0]), np.array([np.ones((2,2))*1])
    left, right, up, down = aggregate_components(y_comp, x_comp, poly = False)
    assert down == 0.0
    assert up == 0.0
    assert left == 0.0
    assert right == 1.0

    y_comp, x_comp = np.array([np.ones((2,2))*0]), np.array([np.ones((2,2))*-1])
    left, right, up, down = aggregate_components(y_comp, x_comp, poly = False)
    assert down == 0.0
    assert up == 0.0
    assert left == 1.0
    assert right == 0.0

    #Down in normal coodinates while input y_comp is in image coordinates (down & up flipped)
    y_comp, x_comp = np.array([np.ones((2,2))*1]), np.array([np.ones((2,2))*0])
    left, right, up, down = aggregate_components(y_comp, x_comp, poly = False)
    assert down == 1.0
    assert up == 0.0
    assert left == 0.0
    assert right == 0.0

    #Up in normal coodinates while input y_comp is in image coordinates (down & up flipped)
    y_comp, x_comp = np.array([np.ones((2,2))*-1]), np.array([np.ones((2,2))*0])
    left, right, up, down = aggregate_components(y_comp, x_comp, poly = False)
    assert down == 0.0
    assert up == 1.0
    assert left == 0.0
    assert right == 0.0

    y_comp, x_comp = np.array([np.ones((2,2))*0]), np.array([np.ones((2,2))*0])
    left, right, up, down = aggregate_components(y_comp, x_comp, poly = False)
    assert down == 0.0
    assert up == 0.0
    assert left == 0.0
    assert right == 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line is the input directory.'
                                        +'The second line is the output directory. All .npy files in the input directory are being processed.')

    parser.add_argument('--debug',  action="store_true")
    parser.add_argument('--figs',  action="store_true")
    args = parser.parse_args()

    if args.debug:
       test_aggregate_flow()
    
    source_folder, target_folder, dataset = get_filenames(args.inputs)
    
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    
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
        process(inpath, target_folder, outfile, dataset, 0, args.figs, args.debug)
