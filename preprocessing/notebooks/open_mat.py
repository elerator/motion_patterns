import numpy as np
import h5py
f = h5py.File('run00.mat','r')
f.keys()
f.get('gcamp')
gcamp = np.array(f.get('gcamp'))
gcamp.shape
hemo = np.array(f.get('hemo'))
