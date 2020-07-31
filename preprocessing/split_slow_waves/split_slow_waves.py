import sys
sys.path.insert(0,'../..')

import xxhash

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pathlib import Path
import argparse
import skimage
from skimage import io

import json

#definiton of variables
source_folder = ""
target_folder = ""
files = []


def where_slow_wave(vector, threshold=.25, dilation = 20):
    up_state = vector > threshold
    up_state = binary_dilation(up_state, iterations=dilation)
    return up_state

def get_slow_wave_start_stop(vector, threshold=.25, dilation = 20):
    where = where_slow_wave(vector, threshold, dilation)
    crossings = list(zero_crossings(where-0.5))
    if where[0]:
        del crossings[0]#beginning already up state
    if where[-1]:
        del crossings[-1]#end up state
    return np.array_split(crossings, len(crossings)//2)


def split_rising_phase(section, smoothing = 1, delay = 0, debug=False):
    derivative = np.gradient(section)

    indices = minima(derivative, smoothing)+delay#Due to smoothing the position of the actual inflection point is shifted; correct using delay
    indices[indices>= len(section)] = len(section)#If delaying beyond vector lengths remove delay
    indices = list(indices)
    indices.append(0)
    indices.append(len(section))
    indices.sort()

    indices = [[i,j] for i,j in zip(indices[:-1], indices[1:]) if j-i > 0]

    if debug:
        plt.plot(section)
        plt.plot(derivative*10)
        print(minima(derivative, smoothing))
        plt.show()

    sections = []
    for start, stop in indices:
        assert start < len(section)
        assert stop <= len(section)
        subsection = section[start:stop]
        assert len(subsection) != 0
        sections.append(subsection)
    #print(sections)
    return indices, sections

n_figure = 0
def subsection_indices(slow_wave, smoothing1 = 10, smoothing2 = 2, delay = 5, debug = True):
    """ Determines indices of sub sections of the slow waves
    Args:
        smoothing1: Smoothing before detecting peaks to distinguish rising and falling phases
        smoothing2: Smoothing applied to derivative before detecting minima that are used to split the rising phase"""
    global n_figure
    section_indices = []
    section_types = []

    minmax = list(maxima(slow_wave, smoothing1))
    minmax.extend(list(minima(slow_wave,smoothing1)))
    minmax.append(0)
    minmax.append(len(slow_wave))
    minmax.sort()

    latest_segement_idx = 0
    if debug:
        fig, ax = plt.subplots(1, dpi=300)

    for start, stop in zip(minmax[:-1], minmax[1:]):
        section = slow_wave[start:stop]

        section_type = ""
        if section[-1] > section[0]:
            section_type = "rising"
        else:
            section_type = "falling"

        if section_type == "rising":
            subsection_indices, subsections = split_rising_phase(section, smoothing2, delay)
            subsection_indices = np.array(subsection_indices) + start
            if debug:
                for [start, stop], sub in zip(subsection_indices, subsections):
                    #print(len(sub))
                    #print(np.arange(stop-start)+start)
                    ax.plot(np.arange(stop-start)+start, sub)
                    assert np.all(slow_wave[start:stop] == sub)
            section_indices.extend(subsection_indices)

            for i in range(len(subsection_indices)):
                latest_segement_idx += 1
                section_types.append(str(latest_segement_idx)+"_rising")
        elif section_type == "falling":
            section_indices.append([start,stop])
            latest_segement_idx += 1
            section_types.append(str(latest_segement_idx)+"_falling")
            if debug:
                ax.plot(np.arange(stop-start)+start,section)
    if debug:
        n_figure += 1
        ax.set_ylabel('df/t')
        ax.set_xlabel("time [ms]")
        fig.savefig('slow_wave'+str(n_figure)+'.png')
        plt.close(fig)

    return np.array(section_indices), section_types

def post_process_slow_wave_start_stops(vector, starts_stops, delay_stop_search_window_size = 100, delay_stop_smoothing= 5, debug=False):
    """ The window for the slow wave starts a little late and stops early sometimes.
        Here we use the local minimum before each slow wave to remove the initial part of the event such that it starts with the rising phase
        The end position is determined as the location where the smooth slope does not have a negative gradient anymore"""

    for i, [start, stop] in enumerate(starts_stops):# Dealy begin to first minimum
        snippet = vector[start:stop]
        try:
            first_minimum = minima(snippet, pre_smoothing = 0)[0]
            starts_stops[i] = [start+first_minimum, stop]
        except:
            print("No minimum found during post_process_slow_wave_start_stops")

    for i, [start, stop] in enumerate(starts_stops):#Delay stop as long as the smooth slope has a negative gradient
        if stop + delay_stop_search_window_size < len(vector):
            snippet = vector[stop:stop+delay_stop_search_window_size]
        else:
            snippet = vector[stop:len(vector)]
        snippet = gaussian_filter(snippet, delay_stop_smoothing)
        if debug:#plot gradient
            fig, ax = plt.subplots(1)
            ax.plot(np.gradient(snippet))
            fig.savefig("test"+str(i)+".png")
            plt.close()
        goes_down = np.gradient(snippet) < 0
        for down in goes_down:
            if down:
                stop += 1
            else:
                break
        starts_stops[i] = [start, stop]

    return starts_stops


def get_slow_wave_segments(vector, min_duration = None, min_auc = 10, smoothing1 = 10, smoothing2 = 2, delay = 5):
    starts_stops = np.array(get_slow_wave_start_stop(vector))#get slow wave positions based on threshold
    starts_stops = post_process_slow_wave_start_stops(vector, starts_stops)#improve detection
    sws = []

    starts_stops_clean = []
    for start_stop in starts_stops:#Add slow waves if slow wave is large enough
        if np.sum(vector[start_stop[0]:start_stop[1]]) >= min_auc:
            sws.append(vector[start_stop[0]:start_stop[1]])
            starts_stops_clean.append(start_stop)
        else:
            print("Area under curve too small, removing slow wave:")
            print(np.sum(vector[start_stop[0]:start_stop[1]]))
    starts_stops = starts_stops_clean

    indices = []#list of lists for each slow wave. Sublist contains sections
    names = []
    for sw, [start, stop] in zip(sws, starts_stops):
        idxs, nms = subsection_indices(sw, smoothing1 = smoothing1, smoothing2 = smoothing2, delay=delay)
        idxs += start
        indices.append(list(idxs))
        names.append(list(nms))

    if min_duration:#
        for i_subsections_list, subsections_list in enumerate(indices):
            for i, [start, stop] in enumerate(subsections_list):
                if stop-start < min_duration:
                    if i+1 < len(subsections_list):
                        if indices[i_subsections_list][i+1][0] < start + min_duration:
                            indices[i_subsections_list][i+1][0] = start + min_duration
                    indices[i_subsections_list][i][1] = start + min_duration
    return starts_stops, indices, names

def perform_splitting(input, target_folder, min_duration=False, smoothing1 = 10, smoothing2 = 2, delay = 5):
    tensor = np.load(input)
    vector = np.nanmean(tensor, axis=(1,2))
    starts_stops, subsection_indices, names = get_slow_wave_segments(vector, min_duration, smoothing1=smoothing1, smoothing2=smoothing2)

    assert len(subsection_indices) == len(names)
    h = xxhash.xxh64()
    h.update(tensor)
    hash = h.hexdigest()
    h.reset()

    slow_wave_folder =  None
    for i, [start_stop, subsections, subsection_names] in enumerate(zip(starts_stops, subsection_indices, names)):
        start, stop = start_stop
        assert len(subsections) == len(subsection_names)
        slow_wave_folder = os.path.join(target_folder, hash + "_" + str(i))
        Path(slow_wave_folder).mkdir(parents=True, exist_ok=True)

        slow_wave_data = tensor[start:stop,:,:].copy()
        #slow_wave_data -= np.nanmin(slow_wave_data)
        #slow_wave_data /= np.nanmax(slow_wave_data)
        np.save(os.path.join(slow_wave_folder, "slow_wave.npy"),slow_wave_data)
        np.save(os.path.join(slow_wave_folder, "slow_wave_mean.npy"), np.nanmean(slow_wave_data, axis=(1,2)))

        fig, ax = plt.subplots(1)
        for subsection, sub_name in zip(subsections, subsection_names):
            start, stop = subsection
            outpath = slow_wave_folder + "/"+ sub_name
            np.save(outpath, tensor[start:stop])
            ax.plot(np.arange(start,stop),np.nanmean(tensor[start:stop], axis=(1,2)))

        fig.savefig(os.path.join(slow_wave_folder, "slow_wave_mean.png"))
        plt.close(fig)
    #print(indices)
    #print(target_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Slow wave segments split')
    parser.add_argument('-inputs', help='Path to config file. The first line must refer to the parent directory.'
                                        +'The second line is the output directory.'
                                        +'Subsequent lines contain the relative path to the files that have to be processed')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('-min_duration', default = 5, type=int)
    parser.add_argument('-smoothing1', default = 10, type=int)
    parser.add_argument('-smoothing2', default = 2, type=int)
    parser.add_argument('-delay', default = 5, type=int)



    args = parser.parse_args()

    try:
        with open(args.inputs) as f:
            source_folder = f.readline()[:-1]
            print("Using source_folder " + source_folder)
            target_folder = f.readline()[:-1]
            print("Using target_folder " + target_folder)
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.rstrip('\n')
                if "#" in line:
                    continue
                files.append(line)
    except Exception as e:
        print("Could not read list of inputs. Make sure you pass the path to a valid file.")
        print(e)
        sys.exit()

    Path(target_folder).mkdir(parents=True, exist_ok=True)

    #Write some metainfo
    metainfo = {"source_folder": source_folder}
    info = json.dumps({**metainfo,**vars(args)}, indent=4)
    with open(os.path.join(target_folder,"metainfo.txt"), "w") as f:
        f.write(info)

    for name in files:
        print("Processing " + str(name))
        input = os.path.join(source_folder,name)
        perform_splitting(input, target_folder, args.min_duration, args.smoothing1, args.smoothing2, args.delay)
