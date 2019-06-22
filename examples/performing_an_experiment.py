import high_dimensional_sampling as hds
import numpy as np

# TODO: Add documentation

class RandomSampling(hds.Method):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        X = np.random.rand(10, len(function.ranges))
        y = function(X).reshape(-1,1)
        return (X,y)
    
    def is_finished(self):
        return False

method = RandomSampling()
function = hds.Easom()
experiment = hds.Experiment(method, '/Users/bstienen/Desktop/hds')

experiment.run(function, finish_line=1000)
