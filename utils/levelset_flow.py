import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion, binary_dilation

from utils.visualization_tools import *

import matplotlib.pyplot as plt

def plot_gradient_field(source, target, quivstep = 5):
    padding = np.max(list(source.shape))//2

    source_outline = source-binary_erosion(source)
    bg = source_outline.copy().astype(np.int32)
    bg += target == 0

    target = np.pad(target, padding)

    dist = distance_transform_edt(target==0)
    dist1 = distance_transform_edt(target==1)
    dist[target>0] = dist1[target>0]
    dist = dist[padding:-padding, padding:-padding]

    fig, ax = plt.subplots(1, figsize=(10,10))
    y_comp, x_comp = np.gradient(dist,2)
    display_combined(-y_comp, -x_comp, bg, fig=fig, ax=ax, scale=10, quivstep=quivstep)
    plt.show()

def grayscale_to_booleans(array, bins):
    output = np.ndarray((bins, array.shape[0],array.shape[1]))
    for i in range(0,bins):
        output[i,:,:] = array >= (i+1)/(bins+1)
    return output

def array_to_vector_field(source, target, vec_y = None, vec_x = None, stepsize=1, smooth=1, max_iterations=1000):
    if type(vec_y) ==  type(None):
        vec_y = np.zeros(source.shape)
        vec_y.fill(np.nan)
    if type(vec_x) == type(None):
        vec_x = np.zeros(target.shape)
        vec_x.fill(np.nan)
    padding = np.max(list(source.shape))//2

    source_outline = source.astype(np.int32) - binary_erosion(source).astype(np.int32)
    target_outline = target.astype(np.int32) - binary_erosion(target).astype(np.int32)

    not_target = target == 0
    not_target = np.logical_or(not_target,target_outline)

    target = np.pad(target, padding)

    dist = distance_transform_edt(target==0)
    dist1 = distance_transform_edt(target==1)
    dist[target>0] = dist1[target>0]
    dist = dist[padding:-padding, padding:-padding]
    grad_y, grad_x = np.gradient(dist,2)

    for y_start, x_start in np.array(np.where(source_outline)).T:
        y = y_start
        x = x_start

        iteration = 0
        #print("---")
        for iteration in range(max_iterations):#While true TODO
            #print(y_comp[int(y),int(x)], end= " ")
            #print(x_comp[int(y),int(x)])
            #print(str(y) + " " + str(x))

            #print(y, end= " ")
            #print(x)

            y -= grad_y[int(round(y)),int(round(x))]*stepsize*2
            x -= grad_x[int(round(y)),int(round(x))]*stepsize*2

            if int(round(y)) < 0 or int(round(x)) < 0:
                break
            if int(round(y)) >= source.shape[0] or int(round(x)) >= source.shape[1]:
                break

            if not_target[int(round(y)),int(round(x))]:
                vec_y[y_start, x_start] = y_start-y
                vec_x[y_start, x_start] = x_start-x

                #print("yep TODO")
                #break

            """print(y, end=" ")
            print(x)
            if not_target[int(round(y)),int(round(x))]:
                img[int(y), int(x)] = 10
                plt.imshow(img)
                plt.show()
                break"""

    if smooth:
        vec_y = smoothen_sparse_array(vec_y, 1)
        vec_x = smoothen_sparse_array(vec_x, 1)

    return -vec_y, -vec_x


def smoothen_sparse_array(array, size=3, proportion=1):
    new = np.ndarray(array.shape)
    new.fill(np.nan)
    for y, x in np.array(np.where(~np.isnan(array))).T:
        snippet = array[y-size  :y+size+1,
                        x-size-1:x+size]
        new_val = np.nanmean(snippet)
        new[y,x] = (proportion * new_val) + (1 - proportion) * array[y,x]
    return new


def levelsets_to_vector_field(levelsets, stepsize):
    """ Take levelsets and compute vector field
    Args:
        source: array of labels
        target: array of labels
    """
    vector_field_shape = levelsets[0][0].shape
    y_comp_combined = np.ndarray(vector_field_shape)
    x_comp_combined = np.ndarray(vector_field_shape)
    y_comp_combined.fill(np.nan)
    x_comp_combined.fill(np.nan)

    for source, target in levelsets:
        labels_present = set(np.array([source.flatten(),target.flatten()]).flatten())
        labels_present.remove(0)#relates to background

        #print(labels_present)
        for l in labels_present:

            source_cluster = source == l
            target_cluster = target == l


            """plt.imshow(source_cluster.astype(np.int32)+target_cluster.astype(np.int32))
            plt.show()
            print("-----------")"""

            #plot_gradient_field(source_cluster.astype(np.int32), target_cluster.astype(np.int32))

            y_comp, x_comp = array_to_vector_field(source_cluster, target_cluster, stepsize=stepsize)
            y_comp_combined[~np.isnan(y_comp)] = y_comp[~np.isnan(y_comp)]
            x_comp_combined[~np.isnan(x_comp)] = x_comp[~np.isnan(x_comp)]
    return y_comp_combined, x_comp_combined


def levelset_flow(x, n_scales=5, stepsize=1):

    y_comps = []
    x_comps = []
    levelsets = []
    for frame1, frame2 in zip(x, x[1:]):
        print(".",end="")
        levelset_source = grayscale_to_booleans(frame1, n_scales)
        levelset_target = grayscale_to_booleans(frame2, n_scales)
        paired = np.array([np.array([levelset1,levelset2]) for levelset1, levelset2 in zip(levelset_source,levelset_target)])
        labeled = np.array([label(x)[0] for x in paired])
        #print(paired.shape)
        #print(labeled.shape)

        y_comp, x_comp = levelsets_to_vector_field(labeled, stepsize=stepsize)
        y_comps.append(y_comp)
        x_comps.append(x_comp)
        levelsets.append(np.mean(levelset_source, axis=0))
    return np.array(y_comps), np.array(x_comps), np.array(levelsets)
    #y_comp, x_comp = levelsets_to_vector_field(source, target)
