import sys
sys.path.insert(0,'../../..')

#import of utilities
from utils.data_transformations import *
from utils.diverse import *
from multiprocessing import Process


#imports
from pathlib import Path
import argparse


def save_spatial_integrals(input_folder, output_folder, subfolder, per_pixel=True, transform_before_integral = None):
	""" Saves spatial integrals
	"""
	integrals_folder = os.path.join(os.path.join(output_folder, "spatial_integrals"), subfolder)
	input_file_names = [f for f in os.listdir(input_folder) if ".npy" in f and not "_mean" in f]
	input_file_paths = [os.path.join(input_folder,f) for f in input_file_names]
	output_file_paths = [os.path.join(integrals_folder, f) for f in input_file_names]

	for f, f_out in zip(input_file_paths, output_file_paths):
		print(".", end= "")
		t = np.load(f)
		if  transform_before_integral:
			t = transform_before_integral(t)
		if per_pixel:
			t = np.nanmean(t, axis=(1,2))
		else:
			t = np.nansum(t, axis=(1,2))
		Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
		np.save(f_out, t)

def save_absolute_frequencies(input_folder, output_folder, subfolder, density=False, transform_before_histogram = None):
	absolute_frequencies_folder = os.path.join(os.path.join(output_folder, "absolute_frequencies"), subfolder)
	bin_boundaries = os.path.join(os.path.join(output_folder, "bin_boundaries"), subfolder)

	input_file_names = [f for f in os.listdir(input_folder) if ".npy" in f and not "_mean" in f]
	input_file_paths = [os.path.join(input_folder,f) for f in input_file_names]
	output_file_paths = [os.path.join(absolute_frequencies_folder, f) for f in input_file_names]
	bin_boundaries_paths = [os.path.join(bin_boundaries, f) for f in input_file_names]

	histograms_temporal, bins_of_hists_temporal = get_histograms(input_file_paths, 1000, density, transform_before_histogram)
		
	for hist, bins, f, f_bins in zip(histograms_temporal, bins_of_hists_temporal,  output_file_paths, bin_boundaries_paths):
		Path(os.path.dirname(f)).mkdir(parents=True, exist_ok=True)
		Path(os.path.dirname(f_bins)).mkdir(parents=True, exist_ok=True)
		np.save(f, np.array(hist))
		np.save(f_bins, np.array(bins))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Preprocessing script')
	parser.add_argument('-inputs', help='Path to config file. The first line is the input directory.'
										+'The second line is the output directory. All .npy files in the input directory are being processed.')

	parser.add_argument('--debug',  action="store_true")
	parser.add_argument('--energy_frequencies',  action="store_true")
	parser.add_argument('--energy_integrals', action="store_true")
	parser.add_argument('--intensity_frequency', action = "store_true")
	parser.add_argument('--intensity_integral', action = "store_true")
	parser.add_argument('--field_strengths_frequency', action = "store_true")
	parser.add_argument('--field_strengths_integral', action = "store_true")
	parser.add_argument('--flow_frequency', action = "store_true")
	parser.add_argument('--flow_integral', action = "store_true")
	parser.add_argument('--sources_integral', action = "store_true")
	parser.add_argument('--sources_frequency', action = "store_true")
	parser.add_argument('--sinks_integral', action = "store_true")
	parser.add_argument('--sinks_frequency', action = "store_true")
	args = parser.parse_args()

	all_metrics = np.all(~np.array([args.energy_frequencies, args.energy_integrals, args.intensity_frequency,
							args.intensity_integral, args.field_strengths_frequency, args.field_strengths_integral,
							args.flow_frequency, args.flow_integral, args.sources_integral, args.sources_frequency,
							args.sinks_integral, args.sinks_frequency]))
	try:
		with open(args.inputs) as f:
			line = f.readline().rstrip('\n')
			contrast_to_downstate_folder = line
			print("Using contrast to downstate folder " + contrast_to_downstate_folder)

			line = f.readline().rstrip('\n')
			field_strengths_folder = line
			print("Using field strengths folder" + field_strengths_folder)

			line = f.readline().rstrip('\n')
			temporal_energy_folder = line
			print("Using temporal energy folder " + temporal_energy_folder)

			line = f.readline().rstrip('\n')
			spatial_energy_folder = line
			print("Using spatial energy folder " + spatial_energy_folder)

			line = f.readline().rstrip('\n')
			source_sink_folder = line
			print("Using source/sink folder " + source_sink_folder)

			line = f.readline().rstrip('\n')
			flow_folder = line
			print("Using flow folder " + flow_folder)

			line = f.readline().rstrip('\n')
			output_folder = line
			print("Using output folder " + output_folder)
	except Exception as e:
		print("Could not read list of inputs. Make sure you pass the path to a valid file.")
		print(e)
	
	if all_metrics or args.energy_frequencies:
		print("\nEnergy frequencies")
		save_absolute_frequencies(temporal_energy_folder, output_folder, "temporal_energy")
		save_absolute_frequencies(spatial_energy_folder, output_folder, "spatial_energy")

	if all_metrics or args.energy_integrals:
		print("\nEnergy integrals")
		save_spatial_integrals(temporal_energy_folder, output_folder, "temporal_energy")
		save_spatial_integrals(spatial_energy_folder, output_folder, "spatial_energy")
			
	if all_metrics or args.intensity_integral:
		print("\nIntensity integrals")
		save_spatial_integrals(contrast_to_downstate_folder, output_folder, "intensity")

	if all_metrics or args.intensity_frequency:
		print("\nIntensity frequency")
		save_absolute_frequencies(contrast_to_downstate_folder, output_folder, "intensity")

	if all_metrics or args.field_strengths_integral:
		print("\nField strengths integral")
		save_spatial_integrals(field_strengths_folder, output_folder, "field_strengths")

	if all_metrics or args.field_strengths_frequency:
		print("\nField strengths frequency")
		save_absolute_frequencies(field_strengths_folder, output_folder, "field_strengths")

	if all_metrics or args.flow_integral:
		print("\nTotal flow")
		save_spatial_integrals(flow_folder, output_folder, "total_flow")

	if all_metrics or args.flow_frequency:
		print("\nFlow distribution")
		save_absolute_frequencies(flow_folder, output_folder, "flow_frequency")

	def only_positive(t):#To set positive/negative to nan before integral/histogram in sources/sinks to select one or the other
		t[t<0] = np.nan
		return t

	def only_negative(t):#To set positive/negative to nan before integral/histogram in sources/sinks to select one or the other
		t[t>0] = np.nan
		return t

	if all_metrics or args.sources_integral:
		print("\nTotal sources")
		save_spatial_integrals(source_sink_folder, output_folder, "sources", transform_before_integral=only_negative)

	if all_metrics or args.sources_frequency:
		print("\nSources distribution")
		save_absolute_frequencies(flow_folder, output_folder, "sources_frequency", transform_before_histogram=only_negative)

	if all_metrics or args.sinks_integral:
		print("\nTotal sinks")
		save_spatial_integrals(source_sink_folder, output_folder, "sinks", transform_before_integral=only_positive)

	if all_metrics or args.sinks_frequency:
		print("\nSinks distribution")
		save_absolute_frequencies(flow_folder, output_folder, "sinks_frequency", transform_before_histogram =only_positive)
