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

import pyscannerbit.scan as sb

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class HdsPsInterface(hds.Procedure):
    def __init__(self, scanner, multinest_tol=0.5, multinest_nlive=100, polychord_tol=1.0, polychord_nlive=20, diver_convthresh=1e-2,  diver_NP=300, twalk_sqrtr=1.05, random_point_number=10000, toy_mcmc_point_number=10):
        self.store_parameters = ['scanner', 'multinest_tol', 'multinest_nlive', 'polychord_tol', 'polychord_nlive', 'diver_convthresh', 'diver_NP', 'twalk_sqrtr', 'random_point_number', 'toy_mcmc_point_number']
        self.scanner = scanner
        self.multinest_tol = multinest_tol
        self.multinest_nlive = multinest_nlive
        self.polychord_tol = polychord_tol
        self.polychord_nlive = polychord_nlive
        self.diver_convthresh = diver_convthresh
        self.diver_NP = diver_NP
        self.twalk_sqrtr = twalk_sqrtr
        self.random_point_number = random_point_number
        self.toy_mcmc_point_number = toy_mcmc_point_number
        self.reset()
    
    def __call__(self, function):
        # Setting for pyscannerbit
        from collections import defaultdict
        def rec_dd():
            return defaultdict(rec_dd)
        settings = rec_dd()
        scan_pars = settings["Scanner"]["scanners"]
        scan_pars["multinest"] = {"tol": self.multinest_tol, "nlive": self.multinest_nlive} 
        scan_pars["polychord"] = {"tol": self.polychord_tol, "nlive": self.polychord_nlive} 
        scan_pars["diver"]     = {"convthresh": self.diver_convthresh, "NP": self.diver_NP} 
        scan_pars["twalk"]     = {"sqrtR": self.twalk_sqrtr}
        scan_pars["random"]    = {"point_number": self.random_point_number}
        scan_pars["toy_mcmc"]  = {"point_number": self.toy_mcmc_point_number} # Acceptance ratio is really bad with this scanner, so don't ask for much

        new_scans = True

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

        myscan = sb.Scan(simple, bounds = ranges, prior_types=["flat"]*dimensions, scanner=self.scanner, settings=settings,fargs=fargs)
        if new_scans:
            print("Running scan with {}".format(self.scanner))
            myscan.scan()
        else:
            print("Retrieving results from previous {} scan".format(self.scanner)) 
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

scanners = ["diver","multinest","polychord","random","toy_mcmc"]

for s in scanners:
    procedure = HdsPsInterface(scanner=s,multinest_tol=0.5,multinest_nlive=100,polychord_tol=1.0,polychord_nlive=20,diver_convthresh=1e-2, diver_NP=300,twalk_sqrtr=1.05,random_point_number=10000,toy_mcmc_point_number=10)
    experiment = hds.OptimisationExperiment(procedure, './hds')
    feeder = hds.functions.FunctionFeeder()
    feeder.load_function('Rastrigin', {'dimensionality': 3})
    #feeder.load_function('Beale')
    for function in feeder:
        experiment.run(function, finish_line=1000) 
