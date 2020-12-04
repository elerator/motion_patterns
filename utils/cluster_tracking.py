import numpy as np
import scipy.stats as st
import copy

class FastDensityClustering():
    @staticmethod
    def gaussian_kernel(size=21, nsig=3):
        """Returns a 2D Gaussian kernel.
        Args:
            size: The size of the kernel (size x size)
            nsig: Sigma of the gaussian
        """
        x = np.linspace(-nsig, nsig, size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()

    @staticmethod
    def density_from_coordinates(coords, shape = [200,200]):
        coords[0] = coords[0]-np.min(coords[0])
        coords[1] = coords[1]-np.min(coords[1])
        coords[0] = coords[0]/np.max(coords[0])
        coords[1] = coords[1]/np.max(coords[1])
        density = np.zeros(shape=shape)
        for y,x in zip(coords[0],coords[1]):
            y*= shape[0]
            x*= shape[1]

            y = int(y)
            x = int(x)

            density[y-1,x-1] += 1
        return density

    @staticmethod
    def kernel(size, ktype):
        """ Returns a kernel of specified size and type
        Args:
            size: Kernel size
            ktype: Type of kernel. Either uniform gaussian or disk are provided.
        """
        if ktype == "uniform":
            return np.ones((size,size))
        elif ktype == "gaussian":
            k = FastDensityClustering.gaussian_kernel(size=size)
            k /= np.max(k)
            return k
        elif ktype == "disk":
            k = FastDensityClustering.gaussian_kernel(size=size)
            k /= np.max(k)
            return k > 0.03

    @staticmethod
    def center_of_mass(matrix):
        weights_x = np.mean(matrix,axis=0)
        weights_y = np.mean(matrix,axis=1)
        if np.sum(weights_x)==0 or np.sum(weights_y)==0:
            return np.array([0,0])

        center_x = np.average(np.arange(len(weights_x)),weights=weights_x)
        center_y = np.average(np.arange(len(weights_y)),weights=weights_y)
        return np.array([center_x,center_y])

    @staticmethod
    def collapse_iteration(arr,kernel, labels=None):
        """ Determins center of gravity for each non-zero (forground pixel) and it's surround weighted by the kernel
            and increases mass at named target position/pixel by the mass of the source pixel.
        Args:
            arr: Grayscale array of positive values where value zero stands for the background and positive values denote the mass for a given foreground pixel.
            kernel: Kernel used to weight the influance of nearby pixels in computing the center of mass
        """
        kernel_width = kernel.shape[0]
        kernel_height = kernel.shape[1]
        ys, xs = np.where(arr>0)
        new = np.zeros(arr.shape)
        abs_shift = 0

        for y, x in zip(ys,xs):
            snippet = arr[y-kernel_width//2:(y+kernel_width//2)+1, x-kernel_width//2:(x+kernel_width//2)+1]

            snippet = kernel * snippet
            shift_x, shift_y = FastDensityClustering.center_of_mass(snippet)
            shift_x -= (kernel_width-1)/2
            shift_y -= (kernel_height-1)/2

            y1 = int(y+shift_y)
            x1 = int(x+shift_x)

            abs_shift += np.abs(shift_x) + np.abs(shift_y)
            new[y1,x1] += arr[y,x]
            if type(labels) != type(None):
                if y1 != y or x1 != x:
                    new_list = []
                    new_list.extend(labels[y1,x1])
                    new_list.extend(labels[y,x])

                    labels[y1,x1] = new_list
                    labels[y,x] = []

        if len(xs)<=0:
            return new, 0, labels
        return new, abs_shift/len(xs), labels

    @staticmethod
    def collapse(arr, iterations = None,gravity_type="uniform", gravity_size=5, labels=True):
        """ Performs clustering by iteratively moving all mass densities (non-zero/foreground pixels) to their center of mass.
        If no value for iterations is specified the algorithm runs until convergence is achieved and the movement is marginally.
        Args:
            arr: Array of positive gray values
            iterations: Number of iterations. If no value for iterations is specified the algorithm runs until convergence is achieved.
            gravity_type: Either "uniform", "gaussian" or "disk". The contributions to the center of mass for one pixels by its surround are weighted accordingly.
            gravity_size: The size of the gravity kernel and the neighborhood used to compute the center of gravity.
        Returns:
            preliminary_cluster_centers: Array representation of the preliminary cluster centers. The array has the same shape as the input arr.
                                         Each cluster center is represented by a non-zero pixel. The value indicates the sum of all densities that converged to it.
            source_locations: An array of lists of points that converged to each location. The array has the same shape as the input arr.
                        The list at the cluster centers contains the orginal coordinates of all points that converged here.

        """
        epsilon = None
        if not iterations:
            iterations = 100000
            epsilon = 1.0e-16

        if gravity_size % 2 == 0:
            gravity_size += 1
        k = FastDensityClustering.kernel(gravity_size,gravity_type)
        arr = np.pad(arr,gravity_size, "constant")

        if not labels:
            labels = None
        else:
            labels = np.ndarray(arr.shape, dtype=object)
            labels.fill([])
            ys, xs = np.where(arr>0)
            for y, x in zip(ys,xs):
                labels[y,x] = [[y-gravity_size,x-gravity_size]]

        for x in range(iterations):
            arr, shift, labels = FastDensityClustering.collapse_iteration(arr,k, labels)
            if epsilon:
                if epsilon > shift:
                    break

        source_locations = np.array(labels[gravity_size:-gravity_size,gravity_size:-gravity_size])
        preliminary_cluster_centers = arr[gravity_size:-gravity_size,gravity_size:-gravity_size]
        return preliminary_cluster_centers, source_locations

    @staticmethod
    def density_clustering(arr, iterations = None,gravity_type="uniform", gravity_size=5):
        """ Performs clustering by iteratively moving all mass densities (non-zero/foreground pixels) to their center of mass.
        If no value for iterations is specified the algorithm runs until convergence is achieved and the movement is marginally.
        Args:
            arr: Array of positive gray values
            iterations: Number of iterations. If no value for iterations is specified the algorithm runs until convergence is achieved.
            gravity_type: Either "uniform", "gaussian" or "disk". The contributions to the center of mass for one pixels by its surround are weighted accordingly.
            gravity_size: The size of the gravity kernel.
        Returns:
            coords_y_component: Coordinates y compontent of all detected cluster centers. The index corresponds to the label of the clusters in cluster_array
            coords_x_component: Coordinates x component of all detected cluster centers. The index corresponds to the label of the clusters in cluster_array
            cluster_densities:
            labels: 3d tensor with labeled slices
        """
        cluster_density_matrix, source_locations = FastDensityClustering.collapse(arr,iterations,gravity_type=gravity_type, gravity_size=gravity_size)
        coords_y_component = []
        coords_x_component = []
        cluster_densities = []

        labeled_array = np.ndarray(arr.shape)
        labeled_array.fill(-1)
        for i,[y,x] in enumerate(np.array(list(np.where(cluster_density_matrix>0))).T):#Go through labels and determine the original coordinates. Determine cluster center as mean of the respective coordinates.
                coords = np.array(source_locations[y,x])
                y_mean, x_mean = coords.mean(axis=0)
                coords_y_component.append(y_mean)
                coords_x_component.append(x_mean)
                labeled_array[coords[:,0],coords[:,1]] = i
                cluster_densities.append(cluster_density_matrix[y,x])
        return np.array(coords_y_component), np.array(coords_x_component), cluster_densities, labeled_array

    @staticmethod
    def closest_label(arr):
        """ Computes location and determins label of the closest pixel to the center location of the array"""
        coords = np.where(arr>=0)
        if len(coords[0]) == 0:
            return None, [-1,-1]
        coords = np.array(coords)
        coords1 = coords-len(arr)//2
        idx = np.argmin(np.square(coords1[0])+np.square(coords1[1]))
        pos = coords[:,idx]

        return arr[pos[0],pos[1]], pos

    @staticmethod
    def assign_closest(coords_t0, mat_t1, search_window_size = 5, unique_descent = True):
        """ Computes mapping between labels based on coordinates (coords_t0) and closest pixels (mat_t1)
        Args:
            coords_t0: Coordinates of clustercenter at time t0 of shape [2,n_coordinates]
            mat_t1: Matrix for time t1 that contains values for the cluster index at the closest position
        """
        search_window_size = (search_window_size//2)*2
        indent = search_window_size#TODO change back
        mat_t1 = np.pad(mat_t1,indent, "constant", constant_values=-1)#TODO change back
        mapping = {}
        mapping_operation = {}

        i = 0
        for y, x in coords_t0.T:#Get snippet/local neighborhood and check for closest label
            snippet = mat_t1[indent+y -search_window_size//2: indent+y+search_window_size+1 -search_window_size//2,
                             indent+x -search_window_size//2: indent+x+search_window_size+1 -search_window_size//2]

            label, pos = FastDensityClustering.closest_label(snippet)

            if type(label) == type(None):
                i += 1
                continue
            #TODO: check for merge

            pos[0] += y+indent-search_window_size//2
            pos[1] += x+indent-search_window_size//2

            mapping[i] = label
            assert label == mat_t1[pos[0],pos[1]]
            mapping_operation[i] = "moved"

            if unique_descent:
                mat_t1[pos[0],pos[1]] = -1#First come first serve. The cluster is not available for other clusters


            i += 1

        return mapping, mapping_operation

    @staticmethod
    def coords_to_label_matrix(matrix_shape, coords):
        """ Puts labels to locations based on matrix of coordinates. This allows fot a quick lookup of the closest center.
        Args:
            matrix_shape: Shape of the output matrix
            coords: Matrix of coordinates where x component is in [:,0] and y component is in [:,1]
        Returns:
            matrix: Matrix with -1 as background value and labes 0...n as foreground values
        """
        mat_t1 = np.ndarray(matrix_shape, dtype=np.int32)
        mat_t1.fill(-1)
        for i, coord in enumerate(coords.T):
            y, x = coord
            mat_t1[y,x] = i

        return mat_t1

    @staticmethod
    def mappings(roi, gravity_size=2, gravity_type="disk", frames = 100, search_window_size=8, mode="closest_center"):
        """ Determines mapping for a tensor of interest and the specified number of frames
        Args:
            roi: A 3D tensor containing the data that will be clustered framewise
            gravity_size: The size of the neighborhood evaluated during clustering to retrieve the center of mass
            gravity_type: The kind and shape of the environment. Either "disk", "gaussian" or "uniform" (for square neighborhood without weighting)
            frames: The number of frames to be processed
            search_window_size: The size of the neighborhood that is evaluated to find the closest clustercenter at t+1
            mode:
        Returns:
            mappings_out: The mappings between labels of clusters in one frame to the ones in the subsequent frame
            preliminary_label_tensor: The tensor of labeled slices. It has the same shape as roi. The background is zero and clusters range from 1 to n.
            coords: The coordinates for the cluster centers of each slice.
        """
        mappings_out = []
        preliminary_label_tensor = []
        coords = []

        res = FastDensityClustering.density_clustering(roi[0], gravity_size=gravity_size, gravity_type=gravity_type)
        frames = np.min([roi.shape[0], frames])
        for frame in range(frames-1):
            print(".",end="")
            #Perform clustering
            res1 = FastDensityClustering.density_clustering(roi[frame+1], gravity_size=gravity_size, gravity_type=gravity_type)

            #Prepare data
            coords_t0 = np.array([res[0],res[1]], dtype=np.int32)
            coords_t1 = np.array([res1[0],res1[1]], dtype = np.int32)
            label_matrix_t1 = FastDensityClustering.coords_to_label_matrix(roi[0].shape,coords_t1)

            #Assign closest
            mapping = None
            if mode == "closest_center_unique":
                mapping, _ = FastDensityClustering.assign_closest(coords_t0, label_matrix_t1, search_window_size=search_window_size, unique_descent = True)
            elif mode == "closest_center":
                mapping, _ = FastDensityClustering.assign_closest(coords_t0, label_matrix_t1, search_window_size=search_window_size, unique_descent = False)

            #print_points_and_background(label_matrix_t1,coords_t0[1],coords_t0[0])
            #plt.show()

            #Accumulate information about clusters
            preliminary_label_tensor.append(res[3])
            mappings_out.append(mapping)
            coords.append(coords_t0)

            res = res1

        return np.array(mappings_out), np.array(preliminary_label_tensor), coords

    @staticmethod
    def rek_track(mapp, key, layer, preliminary_label_tensor, output, final_label):
        if layer>=len(mapp):
            return
        if not key in mapp[layer]:
            return

        output[layer][preliminary_label_tensor[layer]==key] = final_label
        FastDensityClustering.rek_track(mapp, mapp[layer][key], layer+1,preliminary_label_tensor, output, final_label)
        del mapp[layer][key]

    @staticmethod
    def track(mapp, key, preliminary_label_tensor, layer, output, final_label):
        output[layer][preliminary_label_tensor[layer]==key] = final_label
        if key in mapp[layer]:
            FastDensityClustering.rek_track(mapp, mapp[layer][key], layer+1, preliminary_label_tensor, output, final_label)
        else:
            raise Exception("The cluster with the specified label is not in the list of clusters")
        return output

    @staticmethod
    def cluster_tracking(tensor, mapp=None,  mode="closest_center", frames = 100, search_window_size=8):
        """ Assign labels to clusters of each slice such that the same label relates to the same cluster across slices.
            Clusters input tensor framewise and determines mapping between clusters of subsequent frame.
        Args:
            tensor: 3D Tensor where each slice contains clusters of pixels
            mapp: Mapping between clusters per slice
            mode: Mode to determine matching cluster of subsequent frame
            frames: Number of frames to be processed
            search_window_size: Radius of (square) neighborhood where the closest cluster center of the subsequent frame is searched (e.g. in mode closest_center)
        Returns:
            final_tensor: Tensor of labels in which that the same label relates to the same cluster across slices
        """
        mapp1 = None
        if type(mapp) == type(None):
            mapp1, preliminary_label_tensor, coords = FastDensityClustering.mappings(tensor, mode=mode, frames = frames, search_window_size=search_window_size)
        else:
            mapp1 = copy.deepcopy(mapp)
            preliminary_label_tensor = tensor#is that correct?
        i = 0
        final_tensor = None
        out = np.ndarray(preliminary_label_tensor.shape)
        out.fill(-1)
        for layer, layer_dict in enumerate(mapp1):
            #print(len(list(layer_dict.keys())))
            for key in list(layer_dict.keys()):
                final_tensor = FastDensityClustering.track(mapp1, key, preliminary_label_tensor, layer, out, i)
                i += 1

        return final_tensor

    @staticmethod
    def show_trace(tensor, ax):
        """ Plots the pixelwise mean of the tensor """
        #mean = np.zeros((tensor.shape[1],tensor.shape[2]))
        #for x in range(tensor.shape[1]):
        #    for y in range(tensor.shape[2]):
        #        mean[y,x] = np.mean(tensor[:,y,x])
        ax.imshow(np.mean(tensor,axis=0))
        return np.mean(tensor,axis=0)
