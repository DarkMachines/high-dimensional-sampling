from high_dimensional_sampling import methods
from high_dimensional_sampling import functions
from high_dimensional_sampling import experiments
import numpy as np


class RandomSampling(methods.Sampler):
    def __call__(self, function):
        return np.random.rand(10, len(function.ranges))
    
    def is_finished(self):
        return False

method = RandomSampling()
function = functions.Sphere()
experiment = experiments.SamplingExperiment(method, '/Users/bstienen/Desktop/hds')

experiment.run(function, finish_line=1000)
