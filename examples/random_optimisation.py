"""
Example of an optimisation experiment. Implemented procedure is explained at
https://en.wikipedia.org/wiki/Random_optimization
"""
import high_dimensional_sampling as hds
import numpy as np


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
                function.check_ranges(sample, 0.01)
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


procedure = RandomOptimisation(n_initial=5)
experiment = hds.OptimisationExperiment(procedure, '/Users/jdoe/Desktop/hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function_group(['optimisation', 'bounded'])

for function in feeder:
    experiment.run(function, finish_line=1000)
