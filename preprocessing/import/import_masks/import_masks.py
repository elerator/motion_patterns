from scipy.ndimage import gaussian_filter, binary_erosion
import scipy.io
import argparse
from PIL import Image
from pathlib import Path
import numpy as np
import os
def mask_from_roi_labels(roi_labels, dist_between_hemispheres =14, erode=3):
	radius = dist_between_hemispheres//2
	mask = gaussian_filter(np.array(roi_labels>0, dtype=np.float),3)>.5
	mask = binary_erosion(mask, iterations=erode)
	m = np.mean(mask, axis=0)
	centerline = np.argmin(m[len(m)//4:-len(m)//4])+len(m)//4
	mask[:,centerline-radius:centerline+radius] = 0
	return ~mask

def read_config_file(args):
	with open(args.inputs) as f:
		source_path = None
		target_folder = None
		mapping = None
		while True:
			line = f.readline()
			if not line or "#" in line:
				break
			line = line.rstrip('\n')
			if not source_path:
				source_path = line
				continue
			if not target_folder:
				target_folder = line
				continue
			if not mapping:
				mapping = {}   
				sourcelabel, targetname = line.split(" ")
				mapping[sourcelabel] = targetname
	return source_path, target_folder, mapping
			

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Import masks and save as png")
	parser.add_argument("-inputs", help="Path to valid config file")
	args = parser.parse_args()
	source_path, target_folder, mapping = read_config_file(args)
	f = scipy.io.loadmat(source_path)
	Path(target_folder).mkdir(parents=True, exist_ok=True)
	
	for key, target_file_name in mapping.items():#Only one pair in mapping
		mask = mask_from_roi_labels(f[key])
		mask = Image.fromarray(mask.astype(np.uint8)*255)
		mask.save(os.path.join(target_folder, target_file_name))
