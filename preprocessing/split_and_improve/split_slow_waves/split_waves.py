import sys
sys.path.insert(0,'../../..')

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from pygam import LinearGAM
from scipy.spatial import ConvexHull
import re

from scipy.ndimage import minimum_filter
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d


def scan_slow_wave_events(vector, x_minima = None, y_minima = None, x_maxima = None, y_maxima = None, 
                          maximal_height_difference = .5, vector_to_split = None, additional_rule = False):
    if type(x_minima) == type(None):
        x_minima = minima(vector,0)
        y_minima = vector[x_minima]
        x_maxima = maxima(vector, 0)
        y_maxima = vector[x_maxima]
    start = 0#Index of x_minima
    stop = 0
    idx_peak = 0#index of x_maxima
    
    slow_waves = []    
    first_start = None
    
    def find_peak():
        nonlocal idx_peak
        while(x_maxima[idx_peak] < x_minima[start]):#ensure that x of peak > x of min before
            idx_peak += 1
            if idx_peak >= len(x_maxima):#There is no next peak
                break
        if idx_peak >= len(x_maxima):#There is no next peak
            return False
        return True
    
    def find_stop():
        nonlocal stop
        while(x_minima[stop] < x_maxima[idx_peak]):#ensure that x of stop >  x of peak
            stop += 1
            if stop >= len(x_minima):#There is no next potentail stop
                break
        if stop >= len(x_minima):#There is no next potential stop
            return False
        return True
    
    def find_start():
        nonlocal start
        while(x_minima[start] < x_minima[stop]):#The next start value 
            start += 1
            if start >= len(x_minima):#there is no new start value
                break
        if start >= len(x_minima):# there is no new start value
            return False
        return True
          
    while(True):
        if not find_start():
            break
        if not find_peak():
            break
        if not find_stop():
            break
            
        if not first_start:#Remember first startposition of first slow wave
            first_start = x_minima[start]
        
        
        assert x_maxima[idx_peak] > x_minima[start]
        assert x_minima[stop] > x_maxima[idx_peak]
        
        height_at_peak = y_maxima[idx_peak]
        height_at_start = y_minima[start]

        while(True):
            while(True):
                if len(x_maxima) > idx_peak + 1 and x_maxima[idx_peak + 1] < x_minima[stop]:
                    idx_peak +=1#increase idx_peak as long as its x-position is smaller then the stop position
                else:
                    break
                
            height_at_peak = np.max([height_at_peak, y_maxima[idx_peak]])
            height_at_stop = y_minima[stop]

            rel_height = height_at_peak - height_at_start
            rel_height_end_to_start = (height_at_stop -height_at_start)/rel_height

            #print(rel_height_end_to_start)
            if rel_height_end_to_start < maximal_height_difference:
                break
            
            stop += 1#check for next stop if condition holds
            
            if stop >= len(y_minima):
                break
        if stop >= len(y_minima):
            break  

        #plt.plot(vector[x_minima[start]:x_minima[stop]])
        #plt.show()
        #print(height_at_peak-height_at_start)
        if type(vector_to_split) != type(None):
            slow_waves.append(vector_to_split[x_minima[start]:x_minima[stop]])
        else:
            slow_waves.append(vector[x_minima[start]:x_minima[stop]])
    return slow_waves, first_start


def label_slow_waves(sws, start, len_vector):
    """ Returns a label vector of the same length as vector with lables for slow waves starting at zero"""
    labels = np.ndarray(len_vector)
    labels.fill(-1)
    idx = start
    label = 0
    for s in sws:
        labels[idx:idx+len(s)] = label
        idx += len(s)
        label += 1
    return labels

def trim_slow_wave(sw, smoothing = 5, continuous_sequence = True, debug = False):
    smoothing = smoothing * 100 /len(sw)
    smooth = gaussian_filter(sw, smoothing)
    where = smooth < np.max(smooth)* 0.05
     
    original = sw.copy()

    if np.sum(where).astype(int)<2:
        return sw

    x_max = np.argwhere(smooth == np.max(smooth)).flatten()[0]

    if continuous_sequence:
       start = 0#np.argwhere(where[:x_max]).flatten()
       for i in range(x_max):
          if not where[i]:
             break
          start = i
       stop = -1#np.argwhere(where[x_max:]).flatten()
       for i in range(len(where)-x_max):
          if not where[-i]:
              break
          stop = -i
       stop += len(where)

       sw[:start] = None
       sw[stop:] = None
       start = np.argwhere(where[:x_max]).flatten()
       stop = np.argwhere(where[x_max:]).flatten()

    else:
       start = np.argwhere(where[:x_max]).flatten()
       stop = np.argwhere(where[x_max:]).flatten()

       if len(start)>0:
          start = start[-1]
          sw[:start] = None
       if len(stop)>0:
          stop = stop[0] + x_max + 1
          sw[stop:] = None

    if debug and np.nanmax(sw):
       plt.plot(where)
       plt.plot(sw)
       plt.show()

    if np.sum(np.isnan(sw)).astype(int) <= 1:
       sw = original#Trimming failed return original
    return sw

def fit_minima(vector, pre_smoothing= 10, smoothing=None):
    x = [0]
    x.extend(list(minima(vector, pre_smoothing= pre_smoothing)))
    x.append(len(vector)-1)
    y = vector[x]
    if smoothing:
        y = gaussian_filter(y, smoothing)
    return interp1d(x, y)(np.arange(len(vector)))

def fit_baseline(vector, maximal_slope = 0.002, flexibility = 10, fit="spline"):
    min_fit = fit_minima(vector, 10)
    abs_grad = np.abs(np.gradient(min_fit))
    invalid = abs_grad > np.mean(abs_grad)+maximal_slope*np.std(abs_grad)
    min_fit[invalid] = np.nan
    
    if fit == "polyfit":
        fit = np.poly1d(np.polyfit(np.arange(len(min_fit))[~np.isnan(min_fit)],min_fit[~np.isnan(min_fit)], flexibility))(np.arange(len(min_fit)))
    elif fit == "spline":
        gam = LinearGAM(n_splines=flexibility).gridsearch(np.array([np.arange(len(min_fit))[~np.isnan(min_fit)]]).T,min_fit[~np.isnan(min_fit)])
        fit = gam.predict(np.arange(len(min_fit)))
    else:
        raise Exception("Invalid method for fit. Choose polyfit or spline")
    return min_fit, fit

def sarles_bimodality(vector):
    n = len(vector)
    return (skew(vector)**2+1)/(kurtosis(vector)+(3*(n-1)**2/((n-2)*(n-3))))

def fit_baseline_minfilter(vector, minradius = 1000, smooth = 100, ignore_upper_when_bimodal = True, max_slope = .001, n_splines = 25, bimodal_threshold = .6):
    guess = gaussian_filter(minimum_filter(vector,minradius),smooth)
    
    if ignore_upper_when_bimodal and sarles_bimodality(guess) > bimodal_threshold:
        print("bimodal")
        guess[np.abs(np.gradient(guess)) >= max_slope] = None
        guess[guess>np.nanmean(guess)+np.nanstd(guess)] = None
    else:
        guess[np.abs(np.gradient(guess)) >= max_slope] = None
        if not guess[0]:
            guess[0] = vector[0]
        if not guess[-1]:
            guess[-1] = vector[-1]

    x = np.arange(len(guess))[~np.isnan(guess)]
    y = guess[~np.isnan(guess)]
    fit = gaussian_filter(interp1d(x, y, kind="linear")(np.linspace(np.min(x), np.max(x), len(guess))), 10)
    return fit, guess

def correct_baseline_rubberband(vector,k, smoothing = 0):
    convex = vector+ k*np.linspace(0,40, len(vector))**2
    fit = rubberband(np.arange(len(convex)),gaussian_filter(convex, smoothing))
    corrected = convex-fit
    return corrected, fit

def rubberband(x, y):
    v = ConvexHull(np.array(list(zip(x, y)))).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax()]
    return np.interp(x, x[v], y[v])

def detrend_slow_wave(sw):
    first_sample = np.min(np.where(~np.isnan(sw)))
    last_sample = np.max(np.where(~np.isnan(sw)))
    pred = np.poly1d(np.polyfit([first_sample,last_sample], [sw[first_sample],sw[last_sample]],1))(np.arange(len(sw)))
    return sw - pred

def slow_waves_non_continuous(sws, start, vector, mode="vector", min_slope = .02, n_interp = 128, trim = False):
    """ Retrieve a noncontinuous sequence of slow waves
    Args:
        sws: List of slow wave segments
        start: Beginning of first slow wave
        vector: Vector to be split. Must have the same lengths as the original vector that was split
        mode: Either vector or list
        min_slope: Minimal slope required for trimming
        n_interp: Lengths of interpolated sws_normal_interp
        trim: Defines whether or not the beginning and the end of each wave should be trimmed
    Returns:
        sws: Label vector where all positive values dentote a slow wave
        sws_clean: List or vector of cleaned slow wave (i.e. trimmed if desired)
        sws_clean_normal: Normalized wave
        sws_normal_interp: Normalized wave interpolated to have a lengths of n_interp
        len_sws: Lengths of the wave
        height_sws: Height of the wave
    """
    #vector = gaussian_filter(vector, smoothing)
    #sws, start = scan_slow_wave_events(vector, maximal_height_difference = maximal_height_difference)
    #sws = label_slow_waves(sws, start, len(vector))

    sws_clean = []
    sws_clean_normal = []
    sws_normal_interp = []
    len_sws = []
    height_sws = []

    starts = []
    stops = []

    for sw in sws:
        start_rel = 0
        stop_rel = len(sw)#For trimming
        if trim:
           trimmed = trim_slow_wave(sw, min_slope)
        else:
           trimmed = sw

        if mode == "vector":
            sws_clean.extend(trimmed)
            sws_clean_normal.extend(normalize_nan(trimmed))
        elif mode == "list":
            sws_clean.append(trimmed)
            sws_clean_normal.append(normalize_nan(trimmed))
            if trim:
              start_rel = np.min(np.where(~np.isnan(trimmed)))
              stop_rel = np.max(np.where(~np.isnan(trimmed)))-1
              trimmed = trimmed[start_rel:stop_rel]

            len_sws.append(len(trimmed))
            height_sws.append(np.max(trimmed)-np.min(trimmed))

            sws_normal_interp.append(stretch(trimmed, n_interp))
        
        starts.append(start+start_rel)
        stops.append(start+stop_rel)
        start += len(sw)#Start index of non-trimmed slow-wave
    return sws, sws_clean, sws_clean_normal, sws_normal_interp, len_sws, height_sws, starts, stops

#Corresponding hemodynamic signal for each sws
def get_correlations_continuous(start, sws_continuous, hemo_mean):
    hemo_snips = []
    start1 = start
    for s in sws_continuous:
        hemo_snips.append(hemo_mean[start1:start1+len(s)])
        start1 += len(s)
    corrs = []

    for a, b in zip(hemo_snips, sws_continuous):
       mask = ~np.isnan(b)
       a = a[mask]
       b = b[mask]
       assert ~np.any(np.isnan(a))
       assert ~np.any(np.isnan(b))
       pearsons_r = np.corrcoef(a,b)[0,1]
       if np.isnan(pearsons_r):
          print("Pearsons r is None for hemo/gcamp correlation")
       corrs.append(pearsons_r)
    hemo_interp = [stretch(v, 128) for v in hemo_snips]
    return hemo_snips, hemo_interp, corrs
