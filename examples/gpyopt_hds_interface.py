# How to run:
# 1) sudo apt-get install python-pip
# 2) pip install gpyopt

import high_dimensional_sampling as hds
from high_dimensional_sampling.optimisation import GPyOpt


procedure = GPyOpt(initial_design_numdata=5,
                               aquisition_type='EI',
                               exact_feval=True,
                               de_duplication=True,
                               num_cores=2,
                               max_iter=1000,
                               max_time=60,
                               eps=1e-6)
experiment = hds.OptimisationExperiment(procedure, './hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function('Rastrigin', {'dimensionality': 5})
for function in feeder:
    experiment.run(function, finish_line=1000)
