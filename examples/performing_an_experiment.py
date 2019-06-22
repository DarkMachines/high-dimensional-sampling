from high_dimensional_sampling import methods
from high_dimensional_sampling import functions
from high_dimensional_sampling import experiments
import numpy as np

# TODO: store not only X but also outcome y of sampling
# TODO: methodcalls en functioncalls kloppend maken

class RandomSampling(methods.Sampler):
    def __call__(self, function):
        X = np.random.rand(10, len(function.ranges))
        y = function(X)
        return (X,y)
    
    def is_finished(self):
        return False

method = RandomSampling()
function = functions.Easom()
experiment = experiments.SamplingExperiment(method, '/Users/bstienen/Desktop/hds')

experiment.run(function, finish_line=1000)
