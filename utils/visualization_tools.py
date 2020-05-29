import matplotlib.pyplot as plt
import numpy as np

def display_combined(u, v, Inew, scale = 100, quivstep = 3, fig=None, ax=None, figsize=(10,10), vmin =0, vmax=1):
    if not fig:
        fig, ax = plt.subplots(1,figsize=figsize)
    ax.cla()
    im = ax.imshow(Inew, vmin=vmin,vmax=vmax)

    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            ax.arrow(j, i, v[i, j] * scale, u[i, j] * scale, color='red', head_width=0.5, head_length=1,)

    return fig, ax


def print_points_and_background(img, x,y, point_size=10, marker ="."):
    """ Prints samled points in front of image
    Args:
        img: Image as numpy array
        x: Vector of x positions
        y: Vector of y positions

    """
    fig, ax = plt.subplots(1, figsize=(12,10))

    ax.set_xlim((0, img.shape[1]))
    ax.set_ylim((img.shape[0], 0))
    ax.imshow(img==3)
    ax.scatter(x,y, s=point_size,c="red",marker=marker)
