import numpy as np
import high_dimensional_sampling as hds
from string import ascii_lowercase
import itertools
try:
    import pyscannerbit.scan as sb
except ImportError:
    pass
try:
    from mpi4py import MPI
except ImportError:
    pass
try:
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
except NameError:
    pass


class RandomOptimisation(hds.Procedure):
    def __init__(self, n_initial=10, n_sample=10):
        self.store_parameters = ['n_initial', 'n_sample']
        self.n_initial = n_initial
        self.n_sample = n_sample
        self.reset()

    def __call__(self, function):
        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        if self.current_position is None:
            # Initial sampling
            x = self.get_initial_position(ranges, self.n_initial)
            y = function(x)
            i_best = np.argmin(y)
            self.current_position = x[i_best].reshape(1, len(x[i_best]))
            self.current_value = y[i_best].reshape(-1, 1)
            return (self.current_position, self.current_value)
        # Get new point sampled from gaussian
        x = []
        while len(x) < self.n_sample:
            sample = self.get_point(ranges, 1, 1)
            try:
                function.check_ranges(sample)
                x.append(sample)
            except Exception:
                pass
        x = np.array(x).reshape(self.n_sample, -1)
        y = function(x)
        i_best = np.argmin(y)
        if y[i_best] < self.current_value:
            self.current_position = x[i_best].reshape(1, len(x[i_best]))
            self.current_value = y[i_best].reshape(-1, 1)
        return (x[i_best].reshape((1, len(x[0]))), y[i_best].reshape(-1, 1))

    def get_initial_position(self, ranges, n_sample_initial):
        ndim = len(ranges)
        r = np.array(ranges)
        x = np.random.rand(n_sample_initial, ndim)
        x = x * (r[:, 1] - r[:, 0]) + r[:, 0]
        return x

    def get_point(self, ranges, stdev=0.01, n_sample=1):
        cov = np.identity(len(ranges)) * stdev
        return np.random.multivariate_normal(self.current_position[0], cov,
                                             n_sample)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return False

    def check_testfunction(self, function):
        return True


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
