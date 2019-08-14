"""
Plot a 2-dimensional test function. The specific test function can be selected
at the start of this script.
"""
import high_dimensional_sampling as hds
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" CONFIGURATION """
# Function to plot
f = hds.functions.Easom()
# Plot derivative instead of normal function
# Will only work if a derivative is defined in the TestFunction
plot_derivative = False
# Plot the logarithmic value of the functional value
z_axis_logarithmic = False
# Resolution of the plot in terms of number of samples per axis
resolution_x = 1000
resolution_y = 1000


""" SCRIPT """
# Check if dimensionality of function is correct (should be 1)
d = len(f.ranges)
if d is not 2:
    raise Exception(
        "Selected test function has dimensionality {} (2 expected)".format(d))

# Open plot screen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Limit ranges if is ranging to infinity
ranges =  f.ranges
ranges = [[-25, 25],[-25, 25]]
if np.abs(ranges[0][0]) == np.inf:
    for i in range(len(ranges)):
        ranges[i] = [-100, 100]

# Get range to plot
x = np.linspace(ranges[0][0], ranges[0][1], resolution_x)
y = np.linspace(ranges[1][0], ranges[1][1], resolution_y)
x, y = np.meshgrid(x, y)
grid = np.vstack([x.ravel(), y.ravel()]).T

# Get function values and plot them
z = f(grid, plot_derivative)
z = z.reshape(x.shape)
if z_axis_logarithmic:
    z = np.log10(z)
ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=False)

# Finalise plot and show it
plt.grid()
plt.show()
