import pyscannerbit.scan as sb
import pyscannerbit.defaults as defaults
from pyscannerbit.ScannerBit.python import ScannerBit
import matplotlib.pyplot as plt
import copy
import inspect

# Test function
def test_logl(scan,x,y,z):
  scan.print("x+y+z",x+y+z) # Save custom entry to output data
  return -0.5*( (x-10)**2 + (y-15)**2 + (z+3)**2 ) # fitness function to maximise (log-likelihood)

# Prior function; seems not to be optional like it should be so use null function
# def prior(vec, map):
#  pass
# Nope, no good, can't use null function!
# Transform unit hypercube to space of interest manually.
def prior(vec, map):
    map["x"] = 20.0 - 40.0*vec[0]
    map["y"] = 20.0 - 40.0*vec[1]
    map["z"] = 20.0 - 40.0*vec[2]

# Override some scanner settings
# Easier to take the defaults and replace stuff rather than building from scratch
# (although if you do build it from scratch the defaults will be used to fill in
#  any gaps you leave)
settings = copy.deepcopy(defaults._default_options)
#settings["Scanner"]["scanners"]["multinest"] = {"tol": 0.5, "nlive": 500} # Configured for quick and dirty scan 
# Currently cannot use external scanners due to rpath issues. Use internal ones only for now.
settings["Scanner"]["scanners"]["twalk"] = {"sqrtR": 1.05}
#settings["Scanner"]["scanners"]["random"] = {"point_number": 100}
#settings["Scanner"]["scanners"]["toy_mcmc"] = {"point_number": 10}


# Create scan manager object
myscan = sb.Scan(test_logl, bounds=[[1., 40.]] * 3, prior_types=["flat", "flat", "log"], prior_func=prior, scanner="twalk", settings=settings, model_name='model1')
myscan.scan()

# Retrieve h5py group object, augmented with some helpful routines
hdf5 = myscan.get_hdf5()

# Variable names always match the ones used in "loglike"
print(hdf5.get_param_names())

# Best-fit parameters
print(hdf5.get_best_fit("x"))
print(hdf5.get_best_fit("y"))
print(hdf5.get_best_fit("z"))
print(hdf5.get_min_chi_squared())

# np.array of parameter
print(hdf5.get_param("z"))
print(hdf5.get_loglike())

# Plot pairs of parameters
hdf5.make_plot("x", "y")
hdf5.make_plot("x", "LogLike")

# Plot profile likelihood (requires an axis)
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
hdf5.plot_profile_likelihood(ax,"x","y")
ax = fig.add_subplot(122)
hdf5.plot_profile_likelihood(ax,"x","z")
plt.tight_layout()
fig.savefig("scan_object_test_logl.png")

# This is still an HDF5-like object (with the root being the group containing the datasets for this scan) 
# e.g., you can do
print(hdf5["model1::x"])

