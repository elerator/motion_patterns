import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from IPython.display import HTML
from matplotlib import cm
from PIL import Image
from matplotlib import cm
from utils.data_transformations import stretch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def render_arrow_components(up = 1, down=1, left=1, right=1, cmap = "Blues", black_bg = True):
    """ Plots four arrows that point upwards, downwards, leftwards and rightwards
    Args:
        up: Color of upwards arrow (0-1)
        right: Color of rightwards arrow (0-1)
        left: Color of leftwards arrow (0-1)
        down: Color of downwards arrow (0-1)
    Returns:
        RGB array
    """
    fig, ax = plt.subplots(1, figsize=(2,2), facecolor = "black")
    plt.axis("off")
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)

    offset_center = .5
    blue = .54

    epsilon = .001
    up, down, left, right = [v - epsilon for v in [up, down, left, right]]
    colors = [cm.get_cmap(cmap)([up, down, left, right])][0]

    ax.arrow(5, 5+offset_center, 0, 2, width = .5,  color = colors[0])#up
    ax.arrow(5, 5-offset_center, 0, -2, width = .5, color = colors[1])#down
    ax.arrow(5-offset_center, 5, -2, 0, width = .5, color = colors[2])#left
    ax.arrow(5+offset_center, 5, 2, 0, width = .5, color = colors[3])#left

    res = fig2rgb_array(fig)
    plt.close(fig)
    return res

def plot_animation(vectors, i_min = 0, n_frames = 10, norm=True):
    fig, ax = plt.subplots(1)
    p = ax.plot([])[0]
    def plot(i):
        plt.cla()
        for v in vectors:
            v = normalize(v[i_min+i])
            ax.plot(v)
    ani = matplotlib.animation.FuncAnimation(fig, plot, frames=n_frames)
    return ani

def fig2rgb_array(fig):
    """ Converts a matplotlib figure to an rgb array such that it may be displayed as an ImageDisplay
    Args:
        fig: Matplotlib figure
    Returns:
        arr: Image of the plot in the form of a numpy array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def vector_field_animation(y_comp, x_comp, tensor, start = 0, frames = 40, quivstep = 5, scale = 100, figsize = (10,10), arrow_size = 1):
    fig_sim, ax_sim = display_combined(y_comp[0], x_comp[0], tensor[0], figsize = figsize)

    def animate(i, start, x_comp, y_comp, tensor, quivstep, scale):
        i += start
        print(".", end ="")
        display_combined(y_comp[i],x_comp[i], tensor[i+1], fig=fig_sim, ax=ax_sim, scale=scale, quivstep = quivstep)

    ani_sim = matplotlib.animation.FuncAnimation(fig_sim, lambda i: animate(i, start, x_comp, y_comp, tensor, quivstep, scale), frames=frames)
    return ani_sim


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

def show_video(frames, frames1=None, n_frames = 20, startframe=0, orient = "horizontal", figsize=(10,10), vmin=0,vmax=1,cmap="viridis", show_framenumber=True, jshtml=True):
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
        ani = matplotlib.animation.FuncAnimation(fig, lambda i: show_frame(i, im, time_text), frames=n_frames)
        if jshtml:
           return ani.to_jshtml()
        return ani

def superimpose(img, background, cm_background = cm.gray, cm_foreground = cm.viridis):
    background = np.array(Image.fromarray(np.uint8(cm_background(background)*255)))
    img = np.array(Image.fromarray(np.uint8(cm_foreground(img)*255)))
    zero_color = np.uint8(cm_foreground([0,0,0])*255)[0]
    mask = np.where((img == zero_color).all(axis=2))
    img[mask] = background[mask]
    return img


def display_combined(u, v, Inew, scale = 100, quivstep = 3, fig=None, ax=None, figsize=(10,10), vmin =0, vmax=1, head_width=1, mode="minimal", head_length = 1, width = .001, cmap = "viridis"):
    if not fig:
        fig, ax = plt.subplots(1,figsize=figsize)
    ax.cla()
    if mode == "minimal":
       plt.axis("off")
       plt.subplots_adjust(0,0,1,1,0,0)
    im = ax.imshow(Inew, vmin=vmin,vmax=vmax, cmap = cmap)

    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            if np.isnan(v[i,j]) or np.isnan(u[i,j]):
                continue
            ax.arrow(j, i, v[i, j] * scale, u[i, j] * scale, color='red', head_width=head_width, head_length=head_length, width = width)

    return fig, ax

def fig2rgb_array(fig):
    """ Converts a matplotlib figure to an rgb array such that it may be displayed as an ImageDisplay
    Args:
        fig: Matplotlib figure
    Returns:
        arr: Image of the plot in the form of a numpy array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_colored(x, y, ax = None, cmap = 'plasma', set_lims = True):
    """ Plots a colored line using the specified colorbar
    Args:
         x: Train of x values
         y: Train of y values
         ax: Axis of matoliib figure the data is plotted at. If None a figure is created.
         cmap: Colormap
         set_lims: If True the axis limits are set to the minimum and maximum values of the data.
    Returns:
         ax: Matplolib axis
    """
    if type(ax) == type(None):
        fig, ax = plt.subplots(1)

    t = np.linspace(0, 10, len(x))
    def points_to_lc_segements(x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    lc = LineCollection(points_to_lc_segements(x,y), cmap=plt.get_cmap(cmap),norm=plt.Normalize(0, 10))
    lc.set_array(t)
    lc.set_linewidth(3)

    if set_lims:
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
    ax.add_collection(lc)
    return ax

def add_colorbar(ax, vmin = 0, vmax = 1, cmap = "viridis", unit_string = None, fontsize = 10):
    """ Adds colorbar to axis.
    Args:
        ax: Matplotluib axis
        vmin: Miniumum value of colorbar
        vmax: Maximum  value of colorbar
        cmap: Colormap
        unit_string: Axis label for y axis
        fontsize: Fontsize of unit_string
    """
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax= vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=matplotlib.cm.get_cmap(cmap), norm = norm, orientation='vertical')
    if unit_string:
        ax_cb.set_ylabel(unit_string, weight = "bold", fontsize = fontsize)
    plt.gcf().add_axes(ax_cb)

def print_points_and_background(img, x,y, point_size=10, marker ="."):
    """ Prints sampled points in front of image
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
