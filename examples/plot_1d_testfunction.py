"""
Plot a 1-dimensional test function. The specific test function can be selected
at the start of this script.
"""
import high_dimensional_sampling as hds
import numpy as np
import matplotlib.pyplot as plt


""" CONFIGURATION """
# Function to plot
f = hds.functions.BreitWigner()
# Include derivative in the plot if available
plot_derivative = True
# Plot the logarithmic value of the functional value
z_axis_logarithmic = False
# Resolution of the plot in terms of number of samples
resolution = 100000


""" SCRIPT """
# Check if dimensionality of function is correct (should be 1)
d = len(f.ranges)
if d is not 1:
    raise Exception(
        "Selected test function has dimensionality {} (1 expected)".format(d))

# Limit ranges if is ranging to infinity
ranges =  f.get_ranges()
if np.abs(ranges[0][0]) == np.inf:
    for i in range(len(ranges)):
        ranges[i] = [-100, 100]

# Get range to plot
x = np.linspace(ranges[0][0], ranges[0][1], resolution).reshape(-1, 1)

# Get function values and plot them
y = f(x)
plt.plot(x, y, label="Function")

# Check if derivative is defined. If so, plot it
if plot_derivative:
    try:
        yprime = f(x, True)
        plt.plot(x, yprime, label="Derivative")
    except:
        print('x')

# Finalise plot and show it
if z_axis_logarithmic:
    plt.yscale('log')
plt.grid()
plt.legend()
plt.show()
