from scipy.interpolate import interp1d

import numpy as np
import h5py
f = h5py.File('run00.mat','r')
f.keys()
f.get('gcamp')
gcamp = np.array(f.get('gcamp'))
gcamp.shape
hemo = np.array(f.get('hemo'))




def resize_interpolate(tensor, target_lengths=30000,kind="cubic"):
	""" Resizes 3d tensor along first dimension using the specified kind of interpolation. Intended to increase tensor size.
	Args:	
		tensor: Input 3d tensor (numpy ndarray)
		target_lengths: Desired size of output tensor along first axis
		kind: kind of interpolation "cubic" or "linear"
	Returns:
		resized tensor
	"""
	output_tensor = np.ndarray((target_lengths, tensor.shape[1], tensor.shape[2]))
	x_vals = np.arange(len(tensor))
	interlacing_factor = target_lengths//len(tensor)
	n_elem_start = interlacing_factor//2
	n_elem_end = interlacing_factor - n_elem_start

	for y in range(tensor.shape[1]):
		print(".", end="")
		for x in range(tensor.shape[2]):
			y_vals = hemo[:,y,x]
			interpolated = np.ndarray(target_lengths)
			interpolated[:n_elem_start] = [y_vals[0] for x in range(n_elem_start)]#First entries cannot be interpolated
			interpolated[-n_elem_end:] = [y_vals[-1] for x in range(n_elem_end)]
			f = interp1d(x_vals, y_vals, kind=kind)
			interpolated[n_elem_start:-n_elem_end] = f(np.linspace(0,len(tensor)-1,(len(tensor)-1)*interlacing_factor))
			output_tensor[:,y,x] = interpolated
	return output_tensor
			
	
