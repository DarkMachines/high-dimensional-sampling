"""
Example of a posterior sampling experiment. Implemented procedure is explained
at https://en.wikipedia.org/wiki/Rejection_sampling
"""
import high_dimensional_sampling as hds
import numpy as np


class RejectionSampling(hds.Procedure):
    def __init__(self):
        self.store_parameters = []
        self.maximum = None
        self.minimum = None

    def __call__(self, function):
        # Check if maximum is already found otherwise store maximum
        if self.maximum is None:
            self.sample_for_extrama(function)
        # Get point
        found = False
        # Get ranges of the test function. The 0.01 moves the minima 0.01 up
        # and the maxima 0.01 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        while not found:
            x = self.get_point(ranges)
            y = function(x)
            r = np.random.rand()
            s = (y - self.minimum) / (self.maximum - self.minimum)
            if s > r:
                found = True
        return (x, y)

    def get_point(self, ranges, n=1):
        ndim = len(ranges)
        r = np.array(ranges)
        x = np.random.rand(n, ndim)
        x = x * (r[:, 1] - r[:, 0]) + r[:, 0]
        return x

    def sample_for_extrama(self, function):
        ranges = function.get_ranges(0.01)
        print(ranges)
        x = self.get_point(ranges, 100000)
        y = function(x)
        self.maximum = np.max(y)
        self.minimum = np.min(y)

    def reset(self):
        pass

    def is_finished(self):
        return False

    def check_testfunction(self):
        return True


procedure = RejectionSampling()
experiment = hds.PosteriorSamplingExperiment(procedure, './hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function_group(
    'posterior', {
        "Block": {
            "block_size": 8
        },
        "MultivariateNormal": {
            "covariance": [[4, 0], [0, 4]]
        }
    })

for function in feeder:
    experiment.run(function, finish_line=200)
