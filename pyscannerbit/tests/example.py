from sb import Scan, HDF5
from default import settings

# Make a function to investigate

def loglike(x, y, z, extra=False):
  if extra:
      x += 1.
  return -0.5 * ((x - 10.)**2 + (y - 15.)**2 + (z - 30.)**2)
  
 
# Tweak default settings (optional)
 
settings["Scanner"]["scanners"]["de"]["NP"] = 100

# Scan over 1 to 40 for each parameter with "de" scanner, with a combination
# of log and flat priors etc.
  
scan = Scan(loglike, [[1., 40.]] * 3, prior_types=["log", "flat", "log"], kwargs={'extra': False}, scanner="multinest", settings=settings)
scan.scan()

# Extract HDF5 object and look at it

hdf5 = scan.get_hdf5()

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

# This is still an HDF5-like object, e.g., you can do
print(hdf5["/python"]["python_model::x"])
