import sys
sys.path.insert(0,'../..')

import gc

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process

#imports
from pathlib import Path
import argparse

import matplotlib
import matplotlib.animation as animation
matplotlib.rcParams['animation.embed_limit'] = 2**128

def get_filenames(args, output_files_bg = False):
    #Read config file
    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            source_bg = f.readline()[:-1]
            print("Using background folder " + source_bg)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            if output_files_bg:
               target_folder_bg = f.readline()[:-1]
               print("Using target_folder_bg " + target_folder_bg)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()
    input_files = [f for f in os.listdir(source_folder) if ".npy" in f]
    input_files.sort()
    background_paths = [os.path.join(source_bg, f) for f in input_files]
    input_paths = [os.path.join(source_folder, f) for f in input_files]
    output_paths = [os.path.join(target_folder, f.split(".")[0]) for f in input_files]
    if output_files_bg:
        output_paths_bg = [os.path.join(target_folder_bg, f.split(".")[0]) for f in input_files]
        return source_folder, source_bg, target_folder, target_folder_bg, input_paths, background_paths, output_paths, output_paths_bg
    else:
        return source_folder, source_bg, target_folder, input_paths, background_paths, output_paths

def remove_existing_files(filepaths, ending = ""):
    """ Replaces files that already exist by None.
    Args:
        filepaths: List of strings representing filepaths
    Returns:
        List of filepaths where invalid files are None
    """
    valid = []
    for f in filepaths:
         if os.path.isfile(f + ending):
            print(f + " exists already", end = " ")
            print("Delete file if you would like to recompute\n")
            valid.append(None)
         else:
            valid.append(f)
    return valid

def process(input_file, background_file, output_file, output_file_bg, format, vector_scale, fps, quivstep, debug):
    if not output_file:
       return
    y_comp, x_comp = np.load(input_file)
    maxval = np.nanmax([y_comp, x_comp])
    y_comp = y_comp / maxval
    x_comp = x_comp / maxval
    bg = normalize_nan(np.load(background_file))
   
    if debug:
       y_comp = y_comp[:5]
       x_comp = x_comp[:5]
       bg = bg[:6]
       quivstep = 25
    print()
    print(format) 
    anim = vector_field_animation(y_comp, x_comp, bg, frames = len(y_comp), scale = vector_scale, quivstep = quivstep, figsize=(5,5))
    if output_file_bg:
       anim_bg = show_video(bg, n_frames = len(bg), jshtml = False, show_framenumber = True)
    if format == "gif":
       print(len(y_comp))
       anim.save(output_file+".gif", writer='imagemagick', fps=fps)
       if output_file_bg:
          anim_bg.save(output_file_bg+".gif", writer='imagemagick', fps=fps)
    elif format == "mpg":
       Writer = animation.writers['ffmpeg']
       writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
       anim.save(output_file+".mp4", writer=writer)
       if output_file_bg:
          anim_bg.save(output_file_bg+".mp4", writer = writer)
    elif format == "html":
       anim = anim.to_jshtml()
       with open(output_file, "w") as f:
          f.write(anim)
       if output_file_bg:
          anim_bg = anim_bg.to_jshtml()
          with open(output_file_bg, "w") as f:
               f.write(anim_bg)
    plt.close("all")
    del bg
    del y_comp
    del x_comp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--omit_background_animations', action="store_true")
    parser.add_argument('--override', action="store_true")
    parser.add_argument("-format", default="gif", nargs="?", type = str)
    parser.add_argument("-quivstep", default=10, nargs="?", type = int)
    parser.add_argument('-parallel_processes', nargs="?", default=5, type=int)
    parser.add_argument('-vector_scale', nargs="?", default = 20, type=int)
    parser.add_argument('-fps', nargs="?", default = 15, type=int)
    args = parser.parse_args()
    use_default_masks = False


    if args.omit_background_animations:
       source_folder, source_bg, target_folder, input_files, background_files, output_files = get_filenames(args)
       output_files_bg = [None for x in range(len(output_files))]
    else:
       source_folder, source_bg, target_folder, target_folder_bg, input_files, background_files, output_files, output_files_bg = get_filenames(args, True)
       Path(target_folder_bg).mkdir(parents=True, exist_ok=True)
       print("Created " + target_folder_bg)
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    if not args.override:
    	output_files = remove_existing_files(output_files, "." + args.format)

    print("fps: " + str(args.fps))

    processes = []
    max_processes = args.parallel_processes
    n_processes = 0
    for input_file, background_file, output_file, output_file_bg in zip(input_files, background_files, output_files, output_files_bg):
        if args.parallel_processes > 1:
           if n_processes > max_processes:
                 for job in processes:
                     job.join()#wait for all to finish
                 n_processes = 0
           p1 = Process(target=lambda: process(input_file, background_file, output_file, output_file_bg, args.format, args.vector_scale, args.fps, args.quivstep, args.debug))
           p1.start()
           processes.append(p1)
           n_processes += 1
        else:
           print("Single process")
           process(input_file, background_file, output_file, output_file_bg, args.format, args.vector_scale, args.fps, args.quivstep, args.debug)
