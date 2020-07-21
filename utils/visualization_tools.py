import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from IPython.display import HTML
from matplotlib import cm
from PIL import Image

def plot_vector_as_field(y, x, scale=1):
    """ Plots a single vector as a vector field of size (10,10)
    Args:
        y: Y component of vector field
        x: X component of vector field
        scale: Length of vectors
    """
    ty = np.zeros((10,10))
    tx = np.zeros((10,10))

    ty.fill(y)
    tx.fill(x)

    print(np.rad2deg(np.arctan2(ty[0,0],tx[0,0])))
    display_combined(ty, tx, tx, scale=scale)

def show_video(frames, frames1=None, n_frames = 20, startframe=0, orient = "horizontal", figsize=(10,10), vmin=0,vmax=1,cmap="viridis", show_framenumber=True):
    #if len(frames<20):
    #    n_frames = len(frames)

    def show_frames(i, im, im1, time_text):
        im.set_array(frames[i])
        im1.set_array(frames1[i])

    def show_frame(i, im, time_text):
        im.set_array(frames[i])
        if time_text:
            time_text.set_text('time = %.1d' % i)

    if type(frames1) != type(None):
        fig, ax = [None,None]

        if orient == "horizontal":
            fig, ax = plt.subplots(1,2, figsize=figsize)
        elif orient == "vertical":
            fig, ax = plt.subplots(2,1, figsize=figsize)

        time_text = None
        if show_framenumber:
            time_text = plt.figtext(0.5, 0.01, "time " + str(0), ha="center", fontsize=18)

        im = ax[0].imshow(frames[0], cmap = cmap, vmin=vmin,vmax=vmax)
        im1 = ax[1].imshow(frames1[0],vmin=vmin,vmax=vmax)
        ani = matplotlib.animation.FuncAnimation(fig, lambda i: show_frames(i, im, im1, time_text), frames=n_frames).to_jshtml()
        return ani
    else:
        fig, ax = plt.subplots(1, figsize=(10,10))
        time_text = None
        if show_framenumber:
            time_text = plt.figtext(0.5, 0.01, "time " + str(0), ha="center", fontsize=18)

        im = ax.imshow(frames[0], cmap = cmap, vmin=vmin,vmax=vmax)
        ani = matplotlib.animation.FuncAnimation(fig, lambda i: show_frame(i, im, time_text), frames=n_frames).to_jshtml()
        return ani

def superimpose(img, background, cm_background = cm.gray, cm_foreground = cm.viridis):
    background = np.array(Image.fromarray(np.uint8(cm_background(background)*255)))
    img = np.array(Image.fromarray(np.uint8(cm_foreground(img)*255)))
    zero_color = np.uint8(cm_foreground([0,0,0])*255)[0]
    mask = np.where((img == zero_color).all(axis=2))
    img[mask] = background[mask]
    return img


def display_combined(u, v, Inew, scale = 100, quivstep = 3, fig=None, ax=None, figsize=(10,10), vmin =0, vmax=1, head_width=1):
    if not fig:
        fig, ax = plt.subplots(1,figsize=figsize)
    ax.cla()
    im = ax.imshow(Inew, vmin=vmin,vmax=vmax)

    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            if np.isnan(v[i,j]) or np.isnan(u[i,j]):
                continue
            ax.arrow(j, i, v[i, j] * scale, u[i, j] * scale, color='red', head_width=head_width, head_length=1,)

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
