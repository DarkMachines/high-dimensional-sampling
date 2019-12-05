import pyscannerbit.scan as sb
import matplotlib.pyplot as plt

# Test function
def test_logl(x,y,z):
  return -0.5*( (x-10)**2 + (y-15)**2 + (z+3)**2 ) # fitness function to maximise (log-likelihood)

# Override some scanner settings (this is a little ugly for now)
# Can make it a little nicer with defaultdict and some recursion
from collections import defaultdict
def rec_dd():
    return defaultdict(rec_dd)
settings = rec_dd() # Uses a new dict as default value when accessing non-existant keys
settings["Scanner"]["scanners"]["multinest"] = {"tol": 0.5, "nlive": 500} # Configured for quick and dirty scan 

# Create scan manager object
myscan = sb.Scan(test_logl, bounds=[[1., 40.]] * 3, prior_types=["log", "flat", "log"], scanner="multinest", settings=settings)
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
print(hdf5["default::x"])

