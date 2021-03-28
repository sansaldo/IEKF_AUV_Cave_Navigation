import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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

# Plot 2D curves to compare predicted positions with ground truth positions over time (or compare different EKF approaches)
"""
:param x: List or array of shape (# curves, # timesteps) containing the x values for all curves
:param y: List or array of shape (# curves, # timesteps) containing the y values for all curves
:param x_title: x-axis title of the plot
:param y_title: y-axis title of the plot
:param series_names: List of shape (# curves) containing strings for series names to be used in plot legend, e.g., ['gt', 'pred']
:param title: String for the plot title (should tell something about the information in the plot or the estimation approach)
:param save_dir: Directory to save image of plot in
"""
def plot_2d(x, y, x_title, y_title, series_names, title, save_dir=None):
    fig = plt.figure(figsize=(10,10))
    for sx, sy, label in zip(x, y, series_names):
        plt.plot(sx, sy, label=label)
    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
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
"""
def plot_3d(x, y, z, x_title, y_title, z_title, series_names, title, save_dir=None):
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    for sx, sy, sz, label in zip(x, y, z, series_names):
        ax.plot(sx, sy, sz, label=label)
    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    ax.set_zlabel(z_title)
    plt.title(title)
    plt.grid()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, '3d_position_' + '_'.join(series_names) + '.pdf'), bbox_inches = "tight")
    plt.show()