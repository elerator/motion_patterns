import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from pyoptflow import HornSchunck, getimgfiles
from PIL import Image
import os
from scipy.signal import argrelextrema
from skimage import exposure
from skimage.exposure import equalize_adapthist
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.interpolate import interp1d

try:
   from pynhhd import nHHD
except:
   print("Warning. Helmholtz-Decomposition not available")
import sys


def segment_peaks(vector, pre_smoothing_mean = 0, shift = 0):
    """ Determins segments of the signal such that each segement begins and ends with 
	a local mimumum and has exactely one significant peak.
    Args:        
    """
    mins = np.array(minima(normalize_nan(vector), pre_smoothing_mean))
    mins += shift
    mins[mins<1] = 1
    starts = [0]
    starts.extend(mins)
    segments = list(zip(starts, mins))
    return segments

def split_sw_sources(sources, pre_smoothing_mean = 0, segments = None):
    """ Splits sources distribution based on local minima in the mean in time 
    Args:
        sources: Tensor of frames of sources in the optical flow
    Returns:
        seg_means: Means over time of the segments. 2D array.
        max_y: Y coordinate of the argument with maximal intensity (image coordinate)
        max_x: X coordinate of the argument with maxomal intensity
    """
    if not segments:
        mean_sources = np.nanmean(sources, axis = (1,2))
        segments = segment_peaks(mean_sources, pre_smoothing_mean)
    
    seg_means = []
    max_y = []
    max_x = []
    for s in segments:
        mean_img = normalize_nan(np.nanmean(sources[s[0]:s[1]], axis =0))
        seg_means.append(mean_img)

        nama = np.nanmax(mean_img)

        if not np.isnan(nama):
           y, x = np.argwhere(nama == mean_img)[0]
        else:
           y, x = [np.nan, np.nan]

        max_y.append(y)
        max_x.append(x)
    max_y = np.array(max_y)
    max_x = np.array(max_x)
    return seg_means, max_y, max_x

def max_loc_per_frame(tensor):
    """ Retrieves the location of the maximal value per frame in the tensor
    Args:
         tensor: Tensor of frames
    Returns:
         max_y: Y coordinates (Image corrodinates)
         max_x: X coordinates for each frame
    """
    max_y = []
    max_x = []
    for img in tensor:
        y, x = np.argwhere(np.nanmax(img) == img)[0]
        max_y.append(y)
        max_x.append(x)
    return np.array(max_y), np.array(max_x)

def stretch(vector, n_interp, norm = True):
    """ Stretches or squeezes vector to match desired output size """
    res = interp1d(np.arange(len(vector)), vector)(np.linspace(0, len(vector)-1, n_interp))
    if norm:
       res = normalize_nan(res)
    return res

def get_histograms(tensor_files, n_bins = 1000, density=False, transform_before_histogram = None):
    """ Computes histograms for tensor files. The bin boundaries are the same for all files.
        They range from the min to the max of all files.
    Args:
        tensor_files: List of filenames
        bins: Bins of histograms
    Returns:
        hists: List of histograms (absolute frequencies per tensor file)
        bins_of_hists: Bin boundaries of the histograms
    """
    min_vals = []
    max_vals = []

    for file in tensor_files:
        print(".", end= "")
        if transform_before_histogram:
               print("Transforming")
               t = transform_before_histogram(t)
        max_val = np.nanmax(t)
        min_val = np.nanmin(t)
        min_vals.append(min_val)
        max_vals.append(max_val)

    total_min = np.min(min_vals)
    total_max = np.max(max_vals)

    histograms = []
    bins_of_hists = []

    print(" ", end="")

    for file in tensor_files:
        print(".", end= "")
        t = np.load(file)
        hist, bins = np.histogram(t[~np.isnan(t)], range = [total_min, total_max], bins = n_bins, density=density)
        histograms.append(hist)
        bins_of_hists.append(bins)

    return histograms, bins_of_hists

def natural_helmholtz_decompositon(vfield):
    """ Computes the natural Helmholtz decomposition for a single 2D vector field
    Args:
        vfield: A vector field of shape [2, width, height] where the y component comes first.
        This could either be a numpy array or a list [y_component, x_component] where both y_component and x_component refer to 2d numpy arrays.
    Returns:
        solenoidal: Solenoidal vector field (divergence free)
        curl_free: Curl free vector field (divergent field)
        harmonic: Harmonic component vector field
        solenoidal_potential: 2d numpy array
        curl_free_potential: 2d numpy array
        divergence: Divergence of the original vector field. 2d numpy array.
    """
    vfield = [vfield[1], vfield[0]]
    vfield = np.einsum("ijk->kij", vfield)
    vfield = np.einsum("ijk->kij", vfield)

    dims = (vfield.shape[0],vfield.shape[1])
    nhhd = nHHD(grid=dims, spacings=(0.1,0.1))
    nhhd.decompose(vfield)

    solenoidal = np.array([nhhd.r[:,:,1], nhhd.r[:,:,0]])
    curl_free = np.array([nhhd.d[:,:,1], nhhd.d[:,:,0]])
    harmonic = np.array([nhhd.h[:,:,1], nhhd.h[:,:,0]])
    solenoidal_potential = nhhd.nRu
    curl_free_potential = nhhd.nD
    divergence = nhhd.div
    return solenoidal, curl_free, harmonic, solenoidal_potential, curl_free_potential, divergence

def helmholtz_decomposition(vfields, mask = None, approach = "nhhd"):
    """ Computes the helmholtz_decomposition using the specified approach
    args:
        v_fields: Tensor of vector fields of shape [2, n_frames, height, width] or a single vectorfield of shape [2, height, width].
        mask: Mask to be applied on the decompositions
        approach: Approach to compute helmholtz_decomposition. By now only natural helmholtz decomposition is supported.
    returns:
        solenoidal_gradient: Vector field representation of divergence free (solenoidal) component
        solenoidal_function: 2d scalar potential representation of solenoidal_function.
        curl_free_gradient: Vector field representation of curl free component
        curl_free_function: 2d scalar function representation of curl free component. The gradient of this component is the vector field representation (solenoidal_gradient).
    """
    if approach != "nhhd":
        raise NotImplementedException("Only natural helmholtz decomposition is supported by now")

    if vfields.ndim == 3:#Insert empty dimension
        vfields = np.array([[vfields[0]],[vfields[1]]])

    if np.any(np.isnan(vfields[0])):
        print("Replacing nan with zero")
        vfields[np.isnan(vfields)] = 0

    solenoidal = np.zeros(shape=vfields.shape, dtype=np.float32)
    curl_free = np.zeros(shape=vfields.shape, dtype=np.float32)
    curl_free_potential = np.zeros(shape=vfields[0].shape, dtype=np.float32)
    solenoidal_potential = np.zeros(shape=vfields[0].shape, dtype=np.float32)

    for i in range(vfields.shape[1]):
        if i % 10 == 0:
            print(".", end="")
        vfield = vfields[:,i,:,:].copy()
        solenoidal[:,i,:,:], curl_free[:,i,:,:], _, solenoidal_potential[i,:,:], curl_free_potential[i,:,:], _ = natural_helmholtz_decompositon(vfield)
        if type(mask) != type(None):
            solenoidal[0,i][:,:][mask] = np.nan
            solenoidal[1,i][:,:][mask] = np.nan
            curl_free[0,i][:,:][mask] = np.nan
            curl_free[1,i][:,:][mask] = np.nan
            curl_free_potential[i][:,:][mask] = np.nan
            solenoidal_potential[i][:,:][mask] = np.nan

    return solenoidal, solenoidal_potential, curl_free, curl_free_potential

def zero_crossings(vector):
    """ Returns the positions of the zero crossings
    Args:
        vector: Vector with potentially positive and negative values
    Returns:
        zero_crossings: Position of the zero crossings
    """
    return np.where(np.diff(np.sign(vector)))[0]

def horn_schunck(tensor, frames=None):
    if not frames:
        frames = len(tensor)-1
    x_comp = np.ndarray(tensor.shape, dtype=np.double)
    y_comp = np.ndarray(tensor.shape, dtype=np.double)
    tensor[np.isnan(tensor)] = 0#replace nans by zero
    for x in range(frames):
        U, V = HornSchunck(tensor[x,:,:], tensor[x+1,:,:], alpha=1.0, Niter=100)
        x_comp[x] = U
        y_comp[x] = V
        if x % 100 == 0:
           print(".",end="")
           sys.stdout.flush()
    return x_comp[:-1], y_comp[:-1]

def gaussian_filter_nan(U, sigma):
    """ Filters nan in masked images such that the masked are stays of constant size"""
    V=U.copy()
    V[np.isnan(U)]=0
    VV= gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW= gaussian_filter(W,sigma=sigma)

    Z=VV/WW
    Z[np.isnan(U)] = np.nan
    return Z

def normalize_nan(tensor):
    tensor = tensor - np.nanmin(tensor)
    tensor /= np.nanmax(tensor)
    return tensor

def logscale(x):
    """ Transforms data to a logarithmic scale
    Args:
        x: Array-like
    Returns: Log(x+1) for all positive values and -Log(-x) for all negative values within x
    """
    #x = x.copy()
    x[x>0] = np.log(x[x>0]+1)
    x[x<=0] = -np.log(-x[x<=0]+1)
    return x

def logscale_vector_lengths(y_comp, x_comp, iterations=1):
    """ Puts lengths of vectors onto a logscale. Does not affect vector directions.
    Args:
        y_comp: Array of vector field components in y direction
        x_comp: Array of vector field components in x direction
    Returns:
        y_comp: Transformed Y components
        x_comp: Transformed X components
    """
    lengths = np.sqrt(y_comp**2+x_comp**2)
    logscaled = lengths
    for i in range(iterations):
        logscaled = logscale(logscaled)
    factor = logscaled
    factor[lengths!=0] = logscaled[lengths!=0]/lengths[lengths!=0]
    return y_comp*factor, x_comp*factor

def normalize_vector_lengths(y_comp, x_comp):
    """ Normalize lengths of vectors. Does not affect vector directions.
    Args:
        y_comp: Array of vector field components in y direction
        x_comp: Array of vector field components in x direction
    Returns:
        y_comp: Transformed Y components
        x_comp: Transformed X components
    """
    lengths = np.sqrt(y_comp**2+x_comp**2)
    normal = lengths.copy()
    normal[~np.isnan(normal)] = normal[~np.isnan(normal)]
    factor = normal
    factor[lengths!=0] = normal[lengths!=0]/lengths[lengths!=0]
    return y_comp*factor, x_comp*factor


def levelsets(tensor, vmin, vmax, n_sets, hstack = True):
    """ Splits tensor into levelsets.
    Args:
        tensor: 3d tensor of frames
        vmin: Lower bound for clipping. Levelsets are computed such that the smallest levelset starts at vmin.
        vmax: Upper bound for clipping. Levelsets are computed such that the upper threshold for the uppermost levelset is vmax.
        hstack: Defines whether levelsets are stacked vertically or not.
    Returns:
        levelsets
    """
    shape = [n_sets, tensor.shape[0], tensor.shape[1], tensor.shape[2]]
    out = np.ndarray(shape=shape, dtype=np.bool)

    boundaries = np.linspace(vmin, vmax, n_sets+1)[1:]

    for i, x in enumerate(boundaries):
        ten = tensor > x
        out[i,:,:,:] = ten

    if hstack:
        out = np.hstack(out)
        print(out.shape)
        return out


def expected_flow(tensor, n_frames= None):
    """ Computes the optical flow that is expected for a given uniform increase of the brightness between frames.
        For each pair of subsequent frames two surrogate frames are computed (based on the pixelwise mean of both frames)
        such that the mean of the first frame equals the mean of the first original frame and the mean of the second frame equals the mean of the second original frame.
        Optical flow is computed using these surrogate frames.
    Args:
        tensor: 3d Tensor of frames
        n_frames: Number of frames to be evaluated. If argument ins not provided all frames are used.
    Returns:
        Tensor of expected flow.
    """
    if not n_frames:
        n_frames = len(tensor)
    x_comps = []
    y_comps = []
    for i, [frame_0, frame_1] in enumerate(zip(tensor, tensor[1:])):
        if i == n_frames:
            return np.array(y_comps), np.array(x_comps)
        print(".", end="")
        factor = np.mean(frame_1)/np.mean(frame_0)
        mean_of_subsequent = (frame_0+frame_1)/2
        mean_of_subsequent *= np.mean(frame_0)/np.mean(mean_of_subsequent)#make same mean as frame 1

        fake_change = np.array([mean_of_subsequent,mean_of_subsequent *factor])
        x_comp_pred, y_comp_pred = horn_schunck(fake_change)
        x_comps.append(x_comp_pred)
        y_comps.append(y_comp_pred)

def post_process_vector_fields(y_comp, y_comp_expected, x_comp, x_comp_expected, mask, logscale = True, pre_log_factor = 1):
    """ Corrects vector fields by substracting the expected field from the original."""
    y_comp_corrected = [(y_comp[i]-y_comp_expected[i])[0] for i in range(len(y_comp))]
    x_comp_corrected = [(x_comp[i]-x_comp_expected[i])[0] for i in range(len(x_comp))]

    for i, [y, x] in enumerate(zip(y_comp_corrected, x_comp_corrected)):
        y[mask] = np.nan
        x[mask] = np.nan
        if logscale:
            y *= pre_log_factor
            x *= pre_log_factor
            y,x = logscale_vector_lengths(y,x)
        y_comp_corrected[i] = y
        x_comp_corrected[i] = x

    return np.array(y_comp_corrected), np.array(x_comp_corrected)

def framewise_closing(tensor, n_iterations=10, bigdata=False, smoothing = None, post_smoothing=None):
    """ Applies the closing operator (erosion of dilation) for n_iterations.
        Closes gaps between clusters of pixels and removes small clusters.
    Args:
        tensor: Binary tensor
        n_iterations: Number of iterations for erosion and dilation for binary closing.
        post_smoothing: Integer for the strength of smoothing the outline of the binary. Smoothing is achieved using a uniform fillter and binarization.
        bigdata: If true there are side effects but the required amount of data is smaller.
        post_smoothing: Integer for the strength of smoothing the outline of the binary. Smoothing is achieved using a uniform fillter and binarization.
    Returns:
        tensor: Modified data where gaps are closed.
    """
    if not bigdata:
        tensor = tensor.copy()

    for i in range(len(tensor)):
        if i % 10 == 0:
            print(".", end = "")
        array = tensor[i]
        if smoothing:
            array = uniform_filter(array.astype(np.float64), smoothing, output=array)>=1.0

        array = binary_erosion(binary_dilation(array, iterations = n_iterations), iterations = n_iterations)
        if post_smoothing:
            array = uniform_filter(array.astype(np.float64), post_smoothing, output=array)>=1.0

        tensor[i] = array
    return tensor

def poly_smooth_2d(coords, polynomial_degree=11, steps_per_pixel=10, epsilon=.1):
    smooth_x = np.polyfit(np.arange(len(coords[0])), coords[0], deg = polynomial_degree)
    smooth_y = np.polyfit(np.arange(len(coords[1])), coords[1], deg = polynomial_degree)
    poly1 = np.poly1d(smooth_x)
    poly2 = np.poly1d(smooth_y)
    smooth_x = poly1(np.linspace(0,len(coords[0]),len(coords[0])*steps_per_pixel))
    smooth_y = poly2(np.linspace(0,len(coords[1]),len(coords[1])*steps_per_pixel))
    smooth_x[smooth_x<np.min(coords[0])-epsilon] = np.nan
    smooth_x[smooth_x>np.max(coords[0])+epsilon] = np.nan
    smooth_y[smooth_y<np.min(coords[1])-epsilon] = np.nan
    smooth_y[smooth_y>np.max(coords[1])+epsilon] = np.nan

    return [smooth_x,smooth_y]

def upper_decentile_pixels(tensor, percentile=.8, upper_limit = .9):
    """ Clip vlaues such that only values larger then the frame's percentile remain.
        By doing so spacial differences become more clearly visible.
    Args:
        tensor: 3d tensor of frames [frames,:,:]
        percentile: Percentile for lower boundary of clipping
        upper_limit: Values larger then percentile+ upper_limit are clipped
    """
    lower_limit = [np.quantile(img, percentile) for img in tensor]
    upper_limits = [np.quantile(img, upper_limit) for img in tensor]

    percentile_thresholded = tensor.copy()
    for i, [threshold, upper_limit] in enumerate(zip(lower_limit, upper_limits)):
        percentile_thresholded[i][percentile_thresholded[i] < threshold] = threshold
        percentile_thresholded[i][percentile_thresholded[i] > upper_limit] = upper_limit
        percentile_thresholded[i] = normalize(percentile_thresholded[i])
    return percentile_thresholded

def normalize(frames, bigdata=False):
    if not bigdata:
        frames = np.array(frames,dtype=np.float64).copy()
    else:
        frames = np.array(frames,dtype=np.float64)
    frames -= np.min(frames)
    frames /= np.max(frames)
    return frames

def framewise_difference(frames, pixelwise_mean, bigdata=False):
    """ Substracts image from every frame in tensor """
    if not bigdata:
        frames = frames - pixelwise_mean
    else:
        start = len(frames) % 100
        for i in range(start, len(frames), 100):
            #print(".",end="")
            frames[i:i+100] = frames[i:i+100] - pixelwise_mean
        frames[0:start] = frames[0:start] - pixelwise_mean


    return frames

def apply_mask(frames, mask, nan = False):
    for i, f in enumerate(frames):
        if nan:
            f[mask] = np.nan
        else:
            f[mask] = 0
        frames[i] = f
    return frames

def substract_pixel_min(tensor, quantile=None):
    if quantile:
        return tensor - np.quantile(tensor,quantile,axis=0)
    return tensor - np.min(tensor,axis=0)

def min_image(tensor):
    img = np.zeros((tensor.shape[1],tensor.shape[2]))
    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            img[y,x] = np.min(tensor[:,y,x])
    return img


def clipped_adaptive(tensor, clipping=.8):
    tensor = np.array([exposure.equalize_adapthist(normalize(frame)) for frame in tensor])
    if clipping > 0:
        tensor[tensor<clipping] = clipping
    tensor = normalize(tensor)
    return tensor

def sinus(hz = 1, sampling_freq = 100, length = 300):
    x = np.linspace(0,length, sampling_freq*length)
    y = np.sin(x*2*np.pi*hz)
    return [x, y]

def fourier(signal,sampling_rate = 100):
    freq = np.abs(np.fft.fft(signal))
    freq = freq[:len(freq)//2]
    x = np.linspace(0,sampling_rate/2,len(freq))
    return x, freq

from scipy.signal import butter, lfilter
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

def remove_frequency(vector, min, max, sampling_rate = 100, approach="butter", order = 1):
    """ A bandstop filter """

    if approach == "butter":
        filtered = butter_bandstop_filter(vector, min, max, sampling_rate, order)
        return filtered, None, None


    vector_minimum = np.min(vector)
    vector = vector.copy() - vector_minimum#Assert that all values are positive; For some reason this is necessary

    freq = np.fft.fft(vector)
    x = np.linspace(0,sampling_rate*2,len(freq))

    conversion = int((len(vector)/sampling_rate)*2)
    min = min*sampling_rate//conversion
    max = max*sampling_rate//conversion

    min = int(min)
    max = int(max)
    #print(min)
    #print(max)

    freq[min:max] = 0
    freq[len(vector)-max:len(vector)-min] = 0

    filtered = np.abs(np.fft.ifft(freq))
    filtered += vector_minimum#Shift back

    return filtered, x[:len(x)//2], np.abs(freq)[:len(x)//2]

def remove_frequency_from_pixel_vectors(tensor,min, max, approach = "butter"):
    out = np.ndarray(tensor.shape)
    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            out[:,y,x] = remove_frequency(tensor[:,y,x], min, max, approach = approach)[0]
    return out

def lowpass_filter(im, mode = None, keep_fraction = .2, smooth_transition=10):
    """ Lowpass filter that supports filtering in vertical direction (e.g. for horizontal lines) and horizontal direction.
    Filtering is achieved via 2D Fourier Transform and masking out low frequencies based on keep_fraction. A cutoff is avoided by gaussian filtering of the mask.
    Args:
        mode: Either None for filtering in both directions, horizontal or vertical. Note that vertical filters horizontal lines.
        keep_fraction: Value between 0 and 0.5 that defines which proportion of (high) frequencies is to be filtered
        smooth_transition: Smoothing factor used to avoid a strong cutoff
    """
    fft = np.fft.fft2(im)
    r, c = fft.shape
    keep = np.ones((r,c))
    if mode == "vertical" or None:
        keep[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    elif mode == "horizontal" or None:
        keep[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    keep = gaussian_filter(keep, smooth_transition)
    #plt.imshow(keep)
    #plt.colorbar()

    fft = fft*keep
    return np.abs(np.fft.ifft2(fft))

def maxima(vector, pre_smoothing=100, minval=None, maxval = None):
    """ Returns indices of local maxima sorted by value at each indice starting with the highest value
    args:
        vector: Vector of data
        pre_smoothing: Set high values to detect tantial peaks only.
    returns: List of local maxima
    """
    extrema = None
    if pre_smoothing > 0:
        smooth = gaussian_filter(vector,pre_smoothing)
        extrema = argrelextrema(smooth,np.greater)[0]
    else:
        extrema = argrelextrema(vector,np.greater)[0]
    vals = vector[extrema].flatten()
    if minval:
        return extrema[vals>minval]
    if maxval:
        return extrema[vals<maxval]
    assert np.all(extrema < len(vector))
    return extrema

def minima(vector, pre_smoothing=100, minval=None):
    if minval:
        return maxima(-vector, pre_smoothing, maxval = -minval)
    else:
        return maxima(-vector, pre_smoothing)

def filter_frame_blood_vessels(img):
    """ Filters blood vessels from image where they appear dark """
    example = normalize(equalize_adapthist(normalize(img.copy())))
    #example = apply_mask([example], mask)[0]
    #plt.imshow(example)
    #plt.show()
    example = normalize(example-gaussian_filter(example,2))
    #plt.imshow(example)
    #plt.show()

    example = example<np.percentile(example,5)
    #plt.imshow(example)
    #plt.show()
    example = binary_dilation(example)
    example = binary_erosion(example)
    example = binary_erosion(example)

    #plt.imshow(example)
    return example

def filter_blood_vessels(frames, threshold=.05,size_increment=None, n_frames=None):
    if n_frames:
        frames = frames[:n_frames]

    vessels = normalize(np.mean(np.array([filter_frame_blood_vessels(frame) for frame in frames]), axis=0))
    vessels = vessels > threshold
    if size_increment:
        vessels = np.array(vessels, dtype=np.float64)
        vessels = normalize(gaussian_filter(vessels,size_increment))>.1#Increase size
    return vessels

def interpolate_nan_framewise(frames, nan_array, n_frames = None, bigdata=False):
    if not n_frames:
        n_frames = len(frames)
    if not bigdata:
        frames = frames[:n_frames].copy()
        print(".", end="")
    for i in range(len(frames)):
        frames[i][nan_array] = np.nan
    print(".", end="")

    for i in range(0,n_frames):
        if i % 1 == 0:
            print(".", end="")
        frames[i] = interpolate_array(frames[i])
    return frames[:n_frames]


from skimage.util.shape import view_as_windows
def dilate_median(tensor, mask, iterations = 10):
    """ Masks tensor and for each frame it dilates the median of the outline iteratively.
        The remaining background pixels are zero in the output tensor (and not np.nan).
    """
    tensor = apply_mask(tensor, mask, nan=True)
    def get_outline(foreground_mask):
        outline = ~(binary_dilation(foreground_mask)^~foreground_mask)
        return outline
    def dilate_median_frame(frame, mask, n_iterations, inline=False):
        if not inline:
            frame = frame.copy()
        mask = mask.copy()
        for i in range(n_iterations):
            outline = get_outline(~mask)
            frame[outline] = np.nanmedian(np.pad(view_as_windows(frame,(3,3)),((1,1),(1,1),(0,0),(0,0)))[outline],axis=(1,2))
            mask[outline] = False 
        return frame
    
    mask = np.isnan(tensor[0])
    for i, x in enumerate(tensor):
        tensor[i] = dilate_median_frame(x, mask, iterations)
    tensor[np.isnan(tensor)] = 0
    return tensor

def interpolate_array(array, method = "linear"):
    """ Interpolates 2D numpy array """
    array = np.ma.masked_invalid(array)
    x = np.arange(array.shape[1])
    y = np.arange(array.shape[0])
    xx, yy = np.meshgrid(x,y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    interpolated = interpolate.griddata((x1, y1), newarr.ravel(), (xx,yy), method=method)
    return interpolated

def pixelwise_polynomial(tensor, polynomial_degree, ignore_larger = 7):
    output = np.ndarray(tensor.shape)
    len_output_vectors = output.shape[0]
    tensor = tensor[:ignore_larger,:,:].copy()

    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            vector = tensor[:,y,x]
            #plt.plot(np.arange(len(vector)),vector)
            parameters = np.polyfit(np.arange(len(vector)), vector, polynomial_degree)
            output[:,y,x] = np.polyval(parameters, np.arange(len_output_vectors))
    return output

def center_of_mass(a):
    a = a.copy()
    a -= np.min(a)#Needs to be all positive
    columnsum = np.sum(a,axis=0)
    where_columnsum = np.where(columnsum>0)[0]
    rowsum = np.sum(a,axis=1)
    where_rowsum = np.where(rowsum>0)[0]

    y = np.average(where_rowsum, weights=rowsum[where_rowsum])
    x = np.average(where_columnsum, weights=columnsum[where_columnsum])
    return [y,x]

def substract_expected(frames, expected, bigdata=False, mode="median", smooth=True):
    """ Substracts expected image for each frame.
    Args:
        frames: 3D Tensor of frames of shape [n,:,:]
        expected: 3D Tensor of expected frames.
        bigdata: If bigdata is True the original data is being modified.
                 This means the method is not side effect free but more memory efficient.
    Returns:
        frames: Modified frames
    """
    if not bigdata:
        frames = frames.copy()
    if smooth:
        expected = gaussian_filter(expected,2)
    if mode == "mean":
        mean = np.mean(frames,axis=(1,2))#pixelwise mean
        expected_mean = np.mean(expected, axis=(1,2))
    elif mode == "median":
        mean = np.median(frames,axis=(1,2))#pixelwise mean
        expected_mean = np.median(expected, axis=(1,2))
    else:
        raise Exception("No such mode")
    #compute mapping: for each mean it specifies the corresponding index of the expected images
    column_vectors = np.tile(np.array([expected_mean]).transpose(), (1, len(mean)))
    mapping = np.argmin(np.abs(column_vectors - mean), axis=0)
    for i, expected_image_idx in enumerate(mapping):
        frames[i] -= expected[expected_image_idx]
    #frames = normalize(frames)
    #frames -= np.mean(frames, axis=0)
    return frames

def discard_minor_components(tensor, keep_components = 3, n_components=None, pca = None, fit_only = False):
    """ Removes all minor principal components keeping only the first keep_components. Effectively smoothens the data.
    Args:
        tensor: Tensor of frames. Shape [n, :,:]
        n_components: Number of components that are not discarded
    Returns:
        smooth: Reconstructed data based on the first keep_components principal components.
        explained_variance_per_component:
    """
    pca_in = tensor.reshape([tensor.shape[0],tensor.shape[1]*tensor.shape[2]])

    if not n_components:
        n_components = len(tensor)

    if type(pca) == type(None):
        pca = PCA(n_components=n_components)
    decompositions = pca.fit_transform(pca_in)
    if fit_only:
        return pca
    explained_variance_per_component = pca.explained_variance_ratio_.cumsum()
    print(len(explained_variance_per_component))
    decompositions[:,keep_components:] = 0
    smooth = pca.inverse_transform(decompositions).reshape(tensor.shape)
    return smooth, explained_variance_per_component
