import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from constants import cone_times, cone_times_ordered, cone_offsets, cone_offsets_ordered
import numpy as np

# Plot time-series curves to compare predicted values with ground truth over time (or compare different EKF approaches)
"""
:param t: List or array of shape (# timesteps) containing the timesteps for all curves
:param x: List or array of shape (# curves, # timesteps) containing the comparison values for each timestep
:param measurement_name: y-axis title of the plot representing the measurement we're plotting values for
:param series_names: List of shape (# curves) containing strings for series names to be used in plot legend, e.g., ['gt', 'pred']
:param title: String for the plot title (should tell something about the dimension we're comparing values for)
:param save_dir: Directory to save image of plot in
"""
def plot_time_series(t, x, measurement_name, series_names, title, save_dir=None):
    fig = plt.figure(figsize=(10,5))
    for series, label in zip(x, series_names):
        plt.plot(t, series, label=label)
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel(measurement_name)
    plt.title(title)
    plt.grid()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'ts_' + measurement_name + '_' + '_'.join(series_names) + '.pdf'), bbox_inches = "tight")
    plt.show()

cone_colors = {0: 'm', 1: 'y', 2: 'b', 3: 'g', 4: 'r', 5: 'c' }

# Plot 2D curves to compare predicted positions with ground truth positions over time (or compare different EKF approaches)
"""
:param x: List or array of shape (# curves, # timesteps) containing the x values for all curves
:param y: List or array of shape (# curves, # timesteps) containing the y values for all curves
:param x_title: x-axis title of the plot
:param y_title: y-axis title of the plot
:param series_names: List of shape (# curves) containing strings for series names to be used in plot legend, e.g., ['gt', 'pred']
:param title: String for the plot title (should tell something about the information in the plot or the estimation approach)
:param save_dir: Directory to save image of plot in
:param state_times: List of shape (# curves, # timesteps) containing the ROS timestamp for each data point
"""
def plot_2d(x, y, x_title, y_title, series_names, title, save_dir=None, state_times=None):
    fig = plt.figure(figsize=(10,10))
    for sx, sy, label in zip(x, y, series_names):
        plt.plot(sx, sy, label=label)
    
    # Plot estimated cone positions
    if state_times is not None:
        # n_cones = cone_times_flat.shape[0]
        n_cones = cone_times.shape[0]
        for i in range(n_cones):
            for s in range(len(series_names)):
                # cone_time0 = cone_times_flat[i]
                cone_time0 = cone_times[i, 0]
                cone_time1 = cone_times[i, 1]

                # Get belief position on first pass
                x0 = x[s][np.argmin(np.abs(state_times[s] - cone_time0))] + cone_offsets[0,i,0]
                y0 = y[s][np.argmin(np.abs(state_times[s] - cone_time0))] + cone_offsets[0,i,1]

                # And belief position on second pass
                x1 = x[s][np.argmin(np.abs(state_times[s] - cone_time1))] + cone_offsets[1,i,0]
                y1 = y[s][np.argmin(np.abs(state_times[s] - cone_time1))] + cone_offsets[1,i,1]

                # plt.scatter([x0], [y0], c=[cone_colors[s]], marker='^')
                plt.scatter([x0], [y0], c=[cone_colors[i]], marker='^')
                plt.scatter([x1], [y1], c=[cone_colors[i]], marker='^')

    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xlim(-125, 25)
    plt.ylim(-100, 50)
    plt.title(title)
    plt.grid()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, '2d_position_' + '_'.join(series_names) + '.pdf'), bbox_inches = "tight")
    plt.show()

# Plot 3D curves to compare predicted positions with ground truth positions over time (or compare different EKF approaches)
"""
:param x: List or array of shape (# curves, # timesteps) containing the x values for all curves
:param y: List or array of shape (# curves, # timesteps) containing the y values for all curves
:param z: List or array of shape (# curves, # timesteps) containing the z values for all curves
:param x_title: x-axis title of the plot
:param y_title: y-axis title of the plot
:param z_title: z-axis title of the plot
:param series_names: List of shape (# curves) containing strings for series names to be used in plot legend, e.g., ['gt', 'pred']
:param title: String for the plot title (should tell something about the information in the plot or the estimation approach)
:param save_dir: Directory to save image of plot in
:param state_times: Array of shape (# curves, # timesteps) containing the ROS timestamp for each data point
"""
def plot_3d(x, y, z, x_title, y_title, z_title, series_names, title, save_dir=None, state_times=None):
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    for sx, sy, sz, label in zip(x, y, z, series_names):
        ax.plot(sx, sy, sz, label=label)

    # Plot estimated cone positions
    if state_times is not None:
        # n_cones = cone_times_flat.shape[0]
        n_cones = cone_times.shape[0]
        for i in range(n_cones):
            for s in range(len(series_names)):
                # cone_time0 = cone_times_flat[i]
                cone_time0 = cone_times[i, 0]
                cone_time1 = cone_times[i, 1]

                # Get belief position on first pass
                x0 = x[s][np.argmin(np.abs(state_times[s] - cone_time0))]
                y0 = y[s][np.argmin(np.abs(state_times[s] - cone_time0))]
                z0 = z[s][np.argmin(np.abs(state_times[s] - cone_time0))]

                # And belief position on second pass
                x1 = x[s][np.argmin(np.abs(state_times[s] - cone_time1))]
                y1 = y[s][np.argmin(np.abs(state_times[s] - cone_time1))]
                z1 = z[s][np.argmin(np.abs(state_times[s] - cone_time1))]

                # ax.scatter([x0], [y0], zs=[z0], c=[cone_colors[s]], marker='^')
                ax.scatter([x0], [y0], zs=[z0], c=[cone_colors[i]], marker='^')
                ax.scatter([x1], [y1], zs=[z1], c=[cone_colors[i]], marker='^')

    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    ax.set_zlabel(z_title)
    ax.set_zlim()
    plt.title(title)
    plt.grid()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, '3d_position_' + '_'.join(series_names) + '.pdf'), bbox_inches = "tight")
    plt.show()