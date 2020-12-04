from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import string
import sys

sys.path.append("../..")
from utils.diverse import *
from utils.visualization_tools import *
from utils.data_transformations import *
from utils.data_transformations import stretch

def plot_vae_input_output(vae, x, idxs, ax = None):
    if type(ax) == type(None):
       fig, ax = plt.subplots(1, len(idxs))
       if len(idxs) == 1:
           ax = [ax]
    for pos_i,i in enumerate(idxs):
        res = vae.predict(x[i:i+1])
        ax[pos_i].axis("off")
        ax[pos_i].plot(x[i][:-2])
        ax[pos_i].plot(res[0][:-2])

def plot_examples_height_ratio(dataset, threshold = .1, start = 0, n = 10):
    fig, ax = plt.subplots(2)
    g, h, w, corr = slow_wave_features(dataset, ["shape", "height", "width", "correlation"])
    g = (h*g.T).T
    where_is = h/w < threshold
    where_is[h < 5] = False
    where_is[np.abs(corr)<.3] = False
    
    for i in range(start, start+n):
        ax[0].plot(g[where_is][i])

    for i in range(start, start+n):
        ax[1].plot(g[~where_is][i])

    #ax[0].set_ylim((0,50))
    ax[0].set_xlim((0,300))
    #ax[1].set_ylim((0,50))
    ax[1].set_xlim((0,300))

def plot_feature_in_latent_space(encoder, x, feature, ax = None):
    """ Plot color coded featue in latent space as scatter plot.
    Args:
        encoder: Encoder that predicts 2D latent space distribution
        feature: Vector with same length as x
        x: Input to encoder that is mapped to latent space
        ax: If provided data is plotted here
    """
    preds = encoder.predict(x)
    x, y = preds[2].T#v of latent dim
    
    if ax:
        ax.scatter(x, y, c=feature)
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c=feature)
        plt.colorbar()
        plt.show()

import io
def plot_examples_and_boxplot_heights(vae, x_train, iso_train, height_train, idxs = None):
    isos = [1.8, 2.0, 2.2, 2.4, 2.6]
    if not idxs:
       idxs = [[0,11, 21],
	       [0, 8, 22],
	       [0,11, 22],
	       [0,10, 22],
	       [0,10, 30]]
    def plot_examples(isos, idxs):
        fig, ax = plt.subplots(5,3, figsize=(5, 7), dpi=300)
        plt.subplots_adjust(left = .05, right = .95, top = .95, bottom = .05, hspace = 1)

        for i, (iso, idxs) in enumerate(zip(isos, idxs)):
            where = iso_train == iso
            plot_vae_input_output(vae, x_train[where], idxs, ax = ax[i][:3])
        for i in range(5):
            ax[i,1].set_title(str(isos[i]) + "% isopropylene", fontsize=(16))
        im = fig2rgb_array(fig)
        fig.savefig(io.BytesIO())#Prevent showing
        plt.close(fig)
        return im
    fig, ax = plt.subplots(1,2, figsize = (8, 6), dpi = 200)
    plt.subplots_adjust(wspace = .4)

    ax[0].imshow(plot_examples(isos, idxs), aspect="auto")
    ax[0].tick_params(labelbottom=False, labelleft=False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].boxplot([height_train[iso_train == iso] for iso in isos])
    ax[1].set_xticklabels(["1.8%", "2.0%","2.2%", "2.4%", "2.6%"])
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("df/dt")
    ax[1].set_ylabel("df/dt [%]")
    ax[0].set_title("Typical examples")
    ax[1].set_title("Peak amplitude")
    
    for i, a in enumerate(ax):
        a.text(-0.03, 1.03, string.ascii_uppercase[i], transform = a.transAxes, size= 15, weight = "bold")

    from IPython.display import clear_output
    clear_output(wait=True)
    return fig

def plot_slow_wave_dynamics(folder_curl_free_function, dataset, slow_wave_id = "exp_21_run_00_sw_0000", smoothing = 0, plot_sinks = True):
    sources_path = os.path.join(folder_curl_free_function, slow_wave_id+".npy")
    source_sink = np.load(sources_path)
    sources = -source_sink.copy()
    sources[sources < 0] = 0
    
    sinks = source_sink.copy()
    sinks[sinks < 0] = 0
    mean_sources = np.nanmean(sources, axis = (1,2))
    mean_sinks = np.nanmean(sinks, axis = (1,2))
    
    mean_percentage_change = normalize_nan(dataset["sws"][slow_wave_id]["shape"])
    
    if np.all(np.isnan(mean_percentage_change)):
        print("Mean percentage change absent")        
        
    mean_percentage_change = mean_percentage_change[~np.isnan(mean_percentage_change)]
        
    #mean_percentage_change = mean_percentage_change[~np.isnan(mean_percentage_change)]
    print(mean_percentage_change.shape)
    print(sources.shape)
        
    bg = np.nanmean(sources, axis = 0)
    segments = segment_peaks(mean_sources, smoothing, shift = 0)#For both hemispheres
    
    segments = [s for s in segments if s[1]-s[0] > 10]#Minimal duration for each segment
    _, max_y_left, max_x_left = split_sw_sources(sources[:,:,:sources.shape[2]//2], 0, segments = segments)
    _, max_y_right, max_x_right = split_sw_sources(sources[:,:,sources.shape[2]//2:], 0, segments = segments)

    seg_means, _, _ = split_sw_sources(sources[:,:], segments = segments)
    seg_means_sinks, _, _ = split_sw_sources(sinks[:,:], segments = segments)

    # Figure
    if plot_sinks:
    	fig = plt.figure(figsize = (8,6), dpi = 200)
    	gs = GridSpec(3, 4, figure=fig, wspace=.2, hspace = .2, height_ratios = [1, 1.2, .5])
    	ax = []
    	ax.append(fig.add_subplot(gs[0,2:]))
    	ax.append(fig.add_subplot(gs[1,:]))
    	ax.append(fig.add_subplot(gs[0,:2]))
    	ax.append(fig.add_subplot(gs[2,:]))
    else:
    	fig = plt.figure(figsize = (8,6), dpi = 200)
    	gs = GridSpec(2, 4, figure=fig, wspace=.2, hspace = .2, height_ratios = [1, 1.2])
    	ax = []
    	ax.append(fig.add_subplot(gs[0,2:]))
    	ax.append(fig.add_subplot(gs[1,:]))
    	ax.append(fig.add_subplot(gs[0,:2]))

    ax[0].imshow(np.mean(sources, axis = 0), cmap = "gray")

    for i, (max_y, max_x) in enumerate([[max_y_left, max_x_left],[max_y_right, max_x_right]]):        
        if i == 1:
            max_x = max_x + sources.shape[2]//2
        #plot_colored(max_x, max_y, ax[0], set_lims = False)
        ax[0].scatter(max_x, max_y, c = np.arange(len(max_x)), cmap = "plasma", marker="o", s = 75)

    add_colorbar(ax[0],0, len(sources), unit_string = "time [ms]", cmap = "plasma", fontsize =8)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("space x")
    ax[0].set_ylabel("space y")
    ax[0].set_title("Location of peak sources")

    for m in segments:
        ax[2].axvline(m[0], c="lightgray", linestyle = "--")
        ax[2].axvline(m[1], c="lightgray", linestyle = "--")

    ax[2].plot(normalize(mean_sources), label = "Mean sources")
    ax[2].plot(mean_percentage_change, label = "Mean percentage change")
    ax[2].legend(fontsize= 6)
    ax[2].set_title("Segmentation of slow-waves")
    ax[2].set_xlabel("time [ms]")
    ax[2].set_ylabel("normalized signal")
    
    ax[1].imshow(np.hstack(seg_means))
    indent = seg_means[0].shape[1]//2
    _ = ax[1].set_xticks(np.linspace(indent,np.hstack(seg_means).shape[1]-indent, len(seg_means)).astype(int))
    ax[1].set_xticklabels(np.array(segments)[:,0])
    ax[1].set_yticks([])
    ax[1].set_title("Mean distribution of sources for slow-wave segments")
    ax[1].set_xlabel("Segment onset [ms]")


    ax[2].text(-0.03, 1.03, "A", transform = ax[2].transAxes, size= 15, weight = "bold")
    ax[1].text(-0.03, 1.03, "C", transform = ax[1].transAxes, size= 15, weight = "bold")
    ax[0].text(-0.25, 1.03, "B", transform = ax[0].transAxes, size= 15, weight = "bold")


    fontsize_scale = 1/(len(seg_means))
    fontsize_scale += 1
    fontsize_scale /= 2
    fontsize_scale *= 12
    ax[1].tick_params(axis = "x", labelsize = fontsize_scale)

    if plot_sinks:
	    ax[2].plot(normalize(mean_sinks), label = "Mean sinks")
	    ax[3].imshow(np.hstack(seg_means_sinks))
	    _ = ax[3].set_xticks(np.linspace(indent,np.hstack(seg_means).shape[1]-indent, len(seg_means)).astype(int))
	    ax[3].set_xticklabels(np.array(segments)[:,0])
	    ax[3].set_yticks([])
	    ax[3].set_title("Mean distribution of sinks for slow-wave segments")
	    ax[3].set_xlabel("Segment onset [ms]")
	    ax[3].text(-0.03, 1.03, "D", transform = ax[3].transAxes, size= 15, weight = "bold")
	    ax[3].tick_params(axis = "x", labelsize = fontsize_scale)


def plot_latent_space_and_basic_features_for_substantial_waves(manifold_img, iso_train, ratio_vertical_horizontal, up_flow, left_flow, flow_per_auc, width_train, height_train, x_pred_train, y_pred_train, x_range = None, y_range = None, dpi = 100):
    iso_levels = list(set(iso_train))
    iso_levels.sort()
    def format_axes(axes):
        current_iso_level = 0
        for i, ax in enumerate(axes):
            if i == 0:
                continue
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_title("iso " + str(iso_levels[current_iso_level]) + "%")
            if i == 1:
                ax.set_ylabel("z [1]", fontsize = 8)
            ax.set_xlabel("z [0]", fontsize = 8)
            current_iso_level += 1


    fig = plt.figure(figsize=(9, 8), dpi=dpi)

    gs = GridSpec(4, 5, figure=fig, wspace=.2)
    gs.update(wspace=.25, hspace=.5)
    ax = []
    ax.append(fig.add_subplot(gs[:3, :3]))
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

    if type(y_range) == type(None):
        x_min = np.min(x_pred_train)
        y_min = np.min(y_pred_train)
        x_max = np.max(x_pred_train)
        y_max = np.max(y_pred_train)
    else:
        y_min, y_max = y_range
        x_min, x_max = x_range
    
    # Section A
    ax[0].tick_params(bottom=False, top=False, left = False, right=False)
    ax[0].set_title("Latent slow-wave-shape space", fontsize = 10)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].imshow(manifold_img, aspect ="auto")
    ax[0].set_xlabel("z [0]", fontsize = 12)
    ax[0].set_ylabel("z [1]", fontsize = 12)

    #Section B
    ax.append(fig.add_subplot(gs[0:1, 3:4]))
    ax[-1].set_title("Vertical flow [%]", fontsize = 9)
    ax[-1].scatter(x_pred_train, y_pred_train, c=np.sqrt(ratio_vertical_horizontal), cmap="inferno", s= 5*normalize(ratio_vertical_horizontal)+1, marker ="x")
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)

    #Section C
    ax.append(fig.add_subplot(gs[0:1, 4:5]))
    ax[-1].set_title(" Bottom up [%]", fontsize = 9)
    ax[-1].scatter(x_pred_train, y_pred_train, c=np.sqrt(up_flow), cmap="inferno", s= 5*normalize(up_flow)+1, marker ="x")
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)


    #Section D
    ax.append(fig.add_subplot(gs[1:2, 3:4]))
    ax[-1].set_title(" Medial-lateral [%]", fontsize = 9)
    ax[-1].scatter(x_pred_train, y_pred_train, c=np.sqrt(left_flow), cmap="inferno", s= 5*normalize(left_flow)+1, marker ="x")
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)

    #Section E
    ax.append(fig.add_subplot(gs[1:2, 4:5]))
    ax[-1].set_title(" Flow / AUC ", fontsize = 9)
    ax[-1].scatter(x_pred_train, y_pred_train, c=np.sqrt(flow_per_auc), cmap="inferno", s= 5*normalize(flow_per_auc)+1, marker ="x")
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)

    # Section  (Phase)
    ax.append(fig.add_subplot(gs[2, 3]))
    ax[-1].set_title("Phase", fontsize = 10)
    ax[-1].scatter(x_pred_train, y_pred_train, c = width_train, s =5)
    #ax[-1].scatter(x_pred_train[width_train<25], y_pred_train[width_train<25], c = "lightblue", s =5)
    #ax[-1].scatter(x_pred_train[width_train>100], y_pred_train[width_train>100], c = "blue", s =5)
    #ax[-1].scatter(x_pred_train[width_train>150], y_pred_train[width_train>150], c = "yellow", s =5)

    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)
 
    # Section  (Amplitude)
    ax.append(fig.add_subplot(gs[2, 4]))
    ax[-1].set_title("Amplitude", fontsize = 10)
    ax[-1].scatter(x_pred_train, y_pred_train, c = height_train, s =5)
    #ax[-1].scatter(x_pred_train[height_train<5], y_pred_train[height_train<5], c = "lightblue", s =5)
    #ax[-1].scatter(x_pred_train[height_train>4], y_pred_train[height_train>5], c = "blue", s =5)
    #ax[-1].scatter(x_pred_train[height_train>20], y_pred_train[height_train>20], c = "yellow", s =5)

    # Section 
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_xlim((x_min, x_max))
    ax[-1].set_ylim((y_min, y_max))
    ax[-1].set_ylabel("z [1]", fontsize = 8)
    ax[-1].set_xlabel("z [0]", fontsize = 8)
    ax[-1].tick_params(labelbottom=False, labelleft=False)

    for i in range(5):
        ax.append(fig.add_subplot(gs[3:, i]))
        where = iso_train == iso_levels[i]
        xp = x_pred_train[where]
        yp = y_pred_train[where]
        xy = np.vstack([xp,yp])
        z = gaussian_kde(xy)(xy)

        ax[-1].scatter(x_pred_train, y_pred_train, s=10, edgecolor='', c="lightgray")

        ax[-1].scatter(xp, yp, c=z, s=10, edgecolor='', cmap="inferno")
        ax[-1].set_xlim((x_min,x_max))
        ax[-1].set_ylim((y_min,y_max))

    format_axes(ax[-6:])

    #axs[3].scatter(x_pred_train[height_train<5], y_pred_train[height_train<5], c = "yellow", s =5)
    #axs[3].scatter(x_pred_train[height_train>5], y_pred_train[height_train>5], c = "green", s =5)

    for i, a in enumerate(ax):
        if i == 0:
            a.text(-0.03, 1.03, string.ascii_uppercase[i], transform = a.transAxes, size= 15, weight = "bold")
        #elif i == 1:
        #    a.text(-0.06, 1.03, string.ascii_uppercase[i], transform = a.transAxes, size= 15, weight = "bold")
        else:
            a.text(-0.1, 1.1, string.ascii_uppercase[i], transform = a.transAxes, size= 15, weight = "bold")

    #fig.colorbar(a, ax=ax[0], fraction=0.025)
    #fig.suptitle("")
    plt.show()
    return fig
