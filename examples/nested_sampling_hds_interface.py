"""
Example of a posterior sampling experiment. Implemented procedure is explained
at https://en.wikipedia.org/wiki/Rejection_sampling
"""
import high_dimensional_sampling as hds
import numpy as np
from high_dimensional_sampling.posterior.nestedsampling import PolyChord

procedure = PolyChord()
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
