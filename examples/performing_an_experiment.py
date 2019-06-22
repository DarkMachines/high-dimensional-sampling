from high_dimensional_sampling import methods
from high_dimensional_sampling import functions
from high_dimensional_sampling import experiments
import numpy as np

# TODO: Add documentation

class RandomSampling(methods.Method):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        X = np.random.rand(10, len(function.ranges))
        y = function(X).reshape(-1,1)
        return (X,y)
    
    def is_finished(self):
        return False

method = RandomSampling()
function = functions.Easom()
experiment = experiments.Experiment(method, '/Users/bstienen/Desktop/hds')

experiment.run(function, finish_line=1000)
