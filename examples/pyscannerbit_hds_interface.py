# How to run:
# 1) Clone pyscannerbit using: git clone https://github.com/bjfar/pyscannerbit.git
# 2) Navigate to pyscannerbit directory (cd pyscannerbit)
# 3) Run "python setup.py sdist"
# 4) Navigate to 'sdist' folder and record name of tarball
# 5) Return to 'pyscannerbit' folder ("cd ..")
# 6) Run "pip install dist/pyscannerbit-X.X.X.tar.gz", where X is the version number you noted in step 4
# 7) Navigate to 'tests' directory and run "python test_all_scanners.py".
# 8) If successful, congrats. You have successfully installed pyscannerbit. One possible error 
#     previously noted was that yaml-cpp and pybind11 weren't automatically cloning along with 
#     pyscannerbit, which can be fixed by manually cloning them to the provided folders.

import high_dimensional_sampling as hds
from high_dimensional_sampling import results
import numpy as np

from string import ascii_lowercase
import itertools

### PS stuff ###
import pyscannerbit.scan as sb

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

new_scans = True
###


class HdsPsInterface(hds.Procedure):
    def __init__(self):
        self.store_parameters = []
        self.reset()
    
    def __call__(self, function):
        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        ranges = np.array(ranges).tolist()
        
        dimensions = function.get_dimensionality()

        simple = function.get_simple_interface()
        simple.invert(True)

        # Create list of function arguments
        fargs = []
        def iter_all_strings():
            for size in itertools.count(1):
                for t in itertools.product(ascii_lowercase, repeat=size):
                    yield "".join(t)
        for t in itertools.islice(iter_all_strings(), dimensions):
            fargs.append(t)

        myscan = sb.Scan(simple, bounds = ranges, prior_types=["flat"]*dimensions, scanner=s, settings=settings,fargs=fargs)
        if new_scans:
            print("Running scan with {}".format(s))
            myscan.scan()
        else:
            print("Retrieving results from previous {} scan".format(s)) 
        results_ps = myscan.get_hdf5()

        # Create array for sampled parameters
        no_samples = len(results_ps.get_params(fargs[0])[0])
        x = np.zeros((no_samples,dimensions))
        i = 0
        for farg in fargs:
            x[:, i] = results_ps.get_params(farg)[0]
            i = i + 1
            # Print out best values for testing
            print(format(farg), results_ps.get_best_fit(farg))

        # No way to get sampled function values from PyScannerbit, so recalculate
        y = function(x) 
     
        return (x, y)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return True

    def check_testfunction(self, function):
        return True



procedure = HdsPsInterface( )
experiment = hds.OptimisationExperiment(procedure, './hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function('Rastrigin', {'dimensionality': 3})
#feeder.load_function('Beale')

### More PS stuff ###
# Try defining ps variables outside of class declaration
from collections import defaultdict
def rec_dd():
    return defaultdict(rec_dd)
settings = rec_dd()

# Settings for quick and dirty scans. Won't do very well, because the test function is
# actually rather tough!
scan_pars = settings["Scanner"]["scanners"]
scan_pars["multinest"] = {"tol": 0.5, "nlive": 100} 
scan_pars["polychord"] = {"tol": 1.0, "nlive": 20} 
scan_pars["diver"]     = {"convthresh": 1e-2, "NP": 300} 
scan_pars["twalk"]     = {"sqrtR": 1.05}
scan_pars["random"]    = {"point_number": 10000}
scan_pars["toy_mcmc"]  = {"point_number": 10} # Acceptance ratio is really bad with this scanner, so don't ask for much

scanners = ["diver"]

for s in scanners:
    for function in feeder:
        experiment.run(function, finish_line=1000)

df = results.make_dataframe({'simple': './hds'})
