import high_dimensional_sampling as hds
import numpy as np


class RejectionSampling(hds.Method):
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
        while not found:
            x = self.get_point(function.ranges)
            y = function(x)
            r = np.random.rand()
            s = (y-self.minimum)/(self.maximum - self.minimum)
            if s > r:
                found = True
        return (x, y)
    
    def get_point(self, ranges, N=1):
        ndim = len(ranges)
        r = np.array(ranges)
        x = np.random.rand(N, ndim)
        x = x*(r[:,1] - r[:,0]) + r[:,0]
        return x
    
    def sample_for_extrama(self, function):
        x = self.get_point(function.ranges, 100000)
        y = function(x)
        self.maximum = np.max(y)
        self.minimum = np.min(y)

    def is_finished(self):
        return False


method = RejectionSampling()
experiment = hds.Experiment(method, '/Users/bstienen/Desktop/hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_functions(
    'posterior',
    {
        "Block": {
            "block_size": 8
        },
        "MultivariateNormal": {
            "covariance": [[4, 0], [0, 4]]
        }
    }
)

for function in feeder:
    experiment.run(function, finish_line=200)
