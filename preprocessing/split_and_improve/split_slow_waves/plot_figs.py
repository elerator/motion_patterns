import sys
sys.path.insert(0,'../..')

#import of utilities
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.diverse import *

#imports
from scipy.interpolate import interp1d
from split_waves import *


def plot_fit(gcamp_mean, gcamp_mean_raw, fit):
    fig, ax = plt.subplots(2,figsize=(12,8))
    ax[0].plot(gcamp_mean_raw)
    ax[1].plot(gcamp_mean)
    ax[0].plot(fit)
    return fig

def plot_correlation_with_gcamp_per_wave(gcamp_mean, hemo_mean, smoothing_subsegments = 5, height_condition_subsegments = 1, additional_rule = False):
    colors = ['darkblue',"fuchsia",'brown',"purple",'red','green','magenta']
    snippets = gaussian_filter(gcamp_mean, smoothing_subsegments).reshape(4,7500)
    snippets_hemo = normalize(hemo_mean.reshape(4,7500))

    fig, ax = plt.subplots(4, figsize=(12,6), sharex=True)
    plt.tight_layout(h_pad=0)
    plt.subplots_adjust(hspace=0)

    for i, (snip, snip_hemo) in enumerate(zip(snippets, snippets_hemo)):
        if i == 0:
            ax[i].set_ylim(0,1)
        else:
            ax[i].set_ylim(0,.99)
        if i ==3:
            ax[i].set_xlabel("time [ms]")
        if i == 1:
            indent = "                   "
            ax[i].set_ylabel("Normalized signal" + indent)

        sws, start = scan_slow_wave_events(normalize(snip), maximal_height_difference = height_condition_subsegments)
        ax[i].plot(snip_hemo, c = "silver")    
        for sw_no, y_sws in enumerate(sws):
            pearsons_r = np.corrcoef(snip_hemo[start: start+len(y_sws)], y_sws)[0,1]
            if pearsons_r > .6:
                color = "lightblue"
                x = np.arange(start, start+len(y_sws))
                ax[i].fill_between(x,np.zeros(len(x)), np.ones(len(x)), step="pre", alpha=1.0, color=color)
            ax[i].plot(np.arange(start, start+len(y_sws)), y_sws, color = colors[sw_no % 6])        
            start += len(y_sws)
    return fig


def plot_sequence_of_clean_waves(hemo_mean, sws_clean, sws_clean_normal):
    fig, ax = plt.subplots(2, figsize=(20,5))
    ax[0].plot(normalize_nan(sws_clean[:5000]))
    ax[0].plot(normalize(normalize(hemo_mean[:5000])))
    ax[1].plot(normalize_nan(sws_clean_normal[:5000]))
    ax[1].plot(normalize(normalize(hemo_mean[:5000])))
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("Normalized signal")
    ax[1].set_xlabel("time [ms]")
    ax[1].set_ylabel("Normalized signal")
    return fig

def plot_slow_waves_split(gcamp_mean, hemo_mean, smoothing = 0, height_condition = .2, additional_rule = None):
    snippets = gaussian_filter(gcamp_mean, smoothing).reshape(4,7500)
    snippets_hemo = normalize(hemo_mean.reshape(4,7500))

    fig, ax = plt.subplots(4, figsize=(12,6), sharex=True)
    plt.tight_layout(h_pad=0)
    plt.subplots_adjust(hspace=0)

    for i, (snip,snip_hemo) in enumerate(zip(snippets, snippets_hemo)):
        if i == 0:
            ax[i].set_ylim(0,1)
        else:
            ax[i].set_ylim(0,.99)
        if i ==3:
            ax[i].set_xlabel("time [ms]")
        if i == 1:
            indent = "                   "
            ax[i].set_ylabel("Normalized signal" + indent)

        sws, start = scan_slow_wave_events(normalize(snip), maximal_height_difference = height_condition)

        ax[i].plot(snip_hemo, c = "lightgray")    
        for y in sws:
            ax[i].plot(np.arange(start, start+len(y)), y)
            start += len(y)
    return fig


def plot_sample_slow_waves(gcamp_mean, sws, sws_clean):
    fig, ax = plt.subplots(5,5, figsize = (20, 10))
    plt.subplots_adjust(hspace=.5)

    idx = 0
    for i in range(5):
        for j in range(5):
            try:
               sw = gcamp_mean[sws == i*5+j+0]
               ax[i, j].plot(sws_clean[idx])
               ax[i, j].set_ylabel("df/dt [%]")
               ax[i, j].set_xlabel("time [ms]")
               idx += 1
            except:
               continue
    return fig
