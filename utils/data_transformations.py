import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from pyoptflow import HornSchunck, getimgfiles

def horn_schunck(tensor, frames=None):
    if not frames:
        frames = len(tensor)-1
    x_comp = []
    y_comp = []
    for x in range(frames):
        U, V = HornSchunck(tensor[x,:,:], tensor[x+1,:,:], alpha=1.0, Niter=100)
        x_comp.append(U)
        y_comp.append(V)
        print(".",end="")
    return np.array(x_comp), np.array(y_comp)

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

def normalize(frames):
    frames = np.array(frames,dtype=np.float64).copy()
    frames -= np.min(frames)
    frames /= np.max(frames)
    return frames

def framewise_difference(frames, pixelwise_mean, bigdata=False):
    if not bigdata:
        frames = frames - pixelwise_mean
    else:
        #frames = np.array([f-pixelwise_mean for f in frames])
        for i, frame in enumerate(frames):
            if i % 1000 == 0:
                print(".", end = "")
            frames[i] = frame - pixelwise_mean

    return frames

def apply_mask(frames, mask):
    for i, f in enumerate(frames):
        f[mask] = 0
        frames[i] = f
    return frames

def substract_pixel_min(tensor):
    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            tensor[:,y,x] -= np.min(tensor[:,y,x])
    return tensor

def clipped_adaptive(tensor, clipping=.8):
    tensor = np.array([exposure.equalize_adapthist(normalize(frame)) for frame in tensor])
    tensor[tensor<clipping] = clipping
    tensor = normalize(tensor)
    return tensor

def fourier(signal,sampling_rate = 100):
    freq = np.abs(np.fft.fft(signal))
    freq = freq[:len(freq)//2]
    x = np.linspace(0,sampling_rate,len(freq))
    return x, freq

def maxima(vector, pre_smoothing=100, minval=0):
    """ Returns indices of local maxima sorted by value at each indice starting with the highest value
    args:
        vector: Vector of data
        pre_smoothing: Set high values to detect substantial peaks only.
    returns: List of local maxima
    """
    extrema = None
    if pre_smoothing > 0:
        extrema = argrelextrema(gaussian_filter(vector,pre_smoothing),np.greater)[0]
    else:
        extrema = argrelextrema(vector,np.greater)[0]
    vals = vector[extrema].flatten()
    return extrema[vals>minval]
