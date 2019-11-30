# How to run:
# 1) Clone pyscannerbit using:
#  git clone https://github.com/bjfar/pyscannerbit.git
# 2) Navigate to pyscannerbit directory (cd pyscannerbit)
# 3) Run "python setup.py sdist"
# 4) Navigate to 'sdist' folder and record name of tarball
# 5) Return to 'pyscannerbit' folder ("cd ..")
# 6) Run "pip install dist/pyscannerbit-X.X.X.tar.gz",
#  where X is the version number you noted in step 4
# 7) Navigate to 'tests' directory and run "python
#  test_all_scanners.py".
# 8) If successful, congrats. You have successfully installed
#  pyscannerbit. One possible error previously noted was that
#  yaml-cpp and pybind11 weren't automatically cloning along
#  with pyscannerbit, which can be fixed by manually cloning
#  them to the provided folders.

import high_dimensional_sampling as hds
import numpy as np

from string import ascii_lowercase
import itertools

import pyscannerbit.scan as sb

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class HdsPsInterface(hds.Procedure):
    def __init__(self,
                 scanner="badass",
                 multinest_tol=0.5,
                 multinest_nlive=100,
                 polychord_tol=1.0,
                 polychord_nlive=20,
                 diver_convthresh=1e-2,
                 diver_NP=300,
                 twalk_sqrtr=1.05,
                 random_point_number=10000,
                 toy_mcmc_point_number=10,
                 badass_points=1000,
                 badass_jumps=10,
                 pso_NP=400):
        self.store_parameters = ['scanner',
                                 'multinest_tol',
                                 'multinest_nlive',
                                 'polychord_tol',
                                 'polychord_nlive',
                                 'diver_convthresh',
                                 'diver_NP',
                                 'twalk_sqrtr',
                                 'random_point_number',
                                 'toy_mcmc_point_number',
                                 'badass_points',
                                 'badass_jumps',
                                 'pso_NP']
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
        self.badass_points = badass_points
        self.badass_jumps = badass_jumps
        self.pso_NP = pso_NP
        self.reset()

    def __call__(self, function):
        scanner_options = {}
        scanner_options["multinest"] = {"tol": self.multinest_tol,
                                        "nlive": self.multinest_nlive}
        scanner_options["polychord"] = {"tol": self.polychord_tol,
                                        "nlive": self.polychord_nlive}
        scanner_options["diver"] = {"convthresh": self.diver_convthresh,
                                    "NP": self.diver_NP}
        scanner_options["twalk"] = {"sqrtR": self.twalk_sqrtr}
        scanner_options["random"] = {"point_number": self.random_point_number}
        scanner_options["toy_mcmc"] = {"point_number":
                                       self.toy_mcmc_point_number}
        scanner_options["badass"] = {"points": self.badass_points,
                                     "jumps": self.badass_jumps}
        scanner_options["pso"] = {"NP": self.pso_NP}

        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        ranges = np.array(ranges).tolist()

        dimensions = function.get_dimensionality()

        simple = function.get_simple_interface_with_scan()
        simple.invert(True)

        # Create list of function arguments
        fargs = []

        def iter_all_strings():
            for size in itertools.count(1):
                for t in itertools.product(ascii_lowercase, repeat=size):
                    yield "".join(t)
        for t in itertools.islice(iter_all_strings(), dimensions):
            fargs.append(t)

        def prior(vec, map):
            iii = 0
            for argument in fargs:
                map[argument] = (ranges[iii][0]
                                 + (ranges[iii][1]-ranges[iii][0])*vec[iii])
                iii = iii + 1

        myscan = sb.Scan(simple,
                         bounds=ranges,
                         prior_func=prior,
                         prior_types=["flat"]*dimensions,
                         scanner=self.scanner,
                         scanner_options=scanner_options[self.scanner],
                         fargs=fargs)
        print("Running scan with {}".format(self.scanner))
        myscan.scan()
        results_ps = myscan.get_hdf5()

        # Create array for sampled parameters
        no_samples = len(results_ps.get_params(fargs[0])[0])
        x = np.zeros((no_samples, dimensions))
        i = 0
        for farg in fargs:
            x[:, i] = results_ps.get_params(farg)[0]
            i = i + 1
            # Print out best values for testing
            print(format(farg), results_ps.get_best_fit(farg))

        # No way to get sampled function values from PS, so recalculate
        y = function(x)

        return (x, y)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return True

    def check_testfunction(self, function):
        return True


# scanners = ["diver", "multinest", "polychord", "random", "toy_mcmc"]
scanners = ["pso"]

for s in scanners:
    procedure = HdsPsInterface(scanner=s,
                               multinest_tol=0.5,
                               multinest_nlive=100,
                               polychord_tol=1.0,
                               polychord_nlive=20,
                               diver_convthresh=1e-2,
                               diver_NP=300,
                               twalk_sqrtr=1.05,
                               random_point_number=10000,
                               toy_mcmc_point_number=10,
                               badass_points=1000,
                               badass_jumps=10,
                               pso_NP=400)
    experiment = hds.OptimisationExperiment(procedure, './hds')
    feeder = hds.functions.FunctionFeeder()
    feeder.load_function('Rastrigin', {'dimensionality': 3})
    for function in feeder:
        experiment.run(function, finish_line=1000)
