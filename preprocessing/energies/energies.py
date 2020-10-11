import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process


#imports
from pathlib import Path
import argparse

def process(inpath, path_temporal_energy, path_spatial_energy, debug = False):
    t = np.load(inpath)
    #Temporal energy
    temporal_energy = 0.5 * (np.gradient(t, axis=0))**2
    np.save(path_temporal_energy, temporal_energy)
    temporal_energy = None
    
    #Spatial energy
    gradient = np.gradient(t[:], axis= (1,2))
    spatial_energy = 0.5*(gradient[0]**2 + gradient[1]**2)#wurzel wegmachen, confirmen lassen
    gradient = None
    np.save(path_spatial_energy, spatial_energy)

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
            temporal_energy_folder = line
            print("Using temporal energy folder " + temporal_energy_folder)
            line = f.readline().rstrip('\n')
            spatial_energy_folder = line
            print("Using spatial energy folder " + spatial_energy_folder)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    
    Path(spatial_energy_folder).mkdir(parents=True, exist_ok=True)
    Path(temporal_energy_folder).mkdir(parents=True, exist_ok=True)
    
    files = []
    temporal_energy_paths = []
    spatial_energy_paths = []
    for f in os.listdir(source_folder):
        if f.endswith(".npy") and not "_mean" in f:
             outpath  = os.path.join(spatial_energy_folder, f)
             if os.path.isfile(outpath):
                print(f + " exists already", end = " ")
                print("Delete file if you would like to recompute\n")
                continue
             print("Output for spatial energy is "+ outpath)
             spatial_energy_paths.append(outpath)

             outpath  = os.path.join(temporal_energy_folder, f)
             if os.path.isfile(outpath):
                print(f + " exists already", end = " ")
                print("Delete file if you would like to recompute\n")
                continue
             print("Output for spatial energy is "+ outpath)
             temporal_energy_paths.append(outpath)

             files.append(os.path.join(source_folder, f))
    
    processes = []
    for inpath, path_temporal_energy, path_spatial_energy in zip(files, temporal_energy_paths, spatial_energy_paths):
        print("------------------")
        if args.parallel_processes:
           p1 = Process(target=lambda: process(inpath, path_temporal_energy, path_spatial_energy,  args.debug))
           p1.start()
           processes.append(p1)
        else:
           process(inpath, path_temporal_energy, path_spatial_energy, args.debug)
