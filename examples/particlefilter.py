import high_dimensional_sampling as hds
from high_dimensional_sampling.optimisation import ParticleFilter
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import optimisation
import numpy as np
import sys


""" Configure experiment """
# Location to store results
RESULTS_FOLDER = "/scratch/bstienen/hds_particlefilter/results/{}d_wd{}_log{}_sr{}"

input_dimensionality = int(sys.argv[1]) # 2, 3, 5, 7
input_width_decay = float(sys.argv[2]) # 0.9, 0.95, 0.99
input_stdev_configs = bool(int(sys.argv[3])) # 0, 1
input_survival_rates = float(sys.argv[4]) # 0.2, 0.5


# Create functions to loop over
functions = [
    func.HiddenFunction1(input_dimensionality),
    func.HiddenFunction2(input_dimensionality),
    func.HiddenFunction3(input_dimensionality),
    func.HiddenFunction4(input_dimensionality)
]

# Number of seed data points
N_SEED = int(10**(1+np.sqrt(input_dimensionality)))
# Number of samples to add in each iteration
N_ITERATION_STEP = N_SEED*10
# Maximum number of samples to take
MAX_N_SAMPLES = int(1e7)

for function in functions:

    # Create ParticleFilter instance
    procedure = optimisation.ParticleFilter(seed_size=N_SEED,
                                            iteration_size=N_ITERATION_STEP,
                                            boundaries=np.array(function.ranges),
                                            initial_width=1,
                                            wc_decay_rate=input_width_decay,
                                            wc_apply_every_n_iterations=1,
                                            sc_scales_with_boundary=False,
                                            sc_logarithmic=input_stdev_configs,
                                            kc_survival_rate=input_survival_rates)

    # Sample seed and get function values at that seed
    seed_x = np.random.rand(N_SEED, input_dimensionality)
    for i, d in enumerate(function.ranges):
        width = d[1] - d[0]
        seed_x[:, i] = seed_x[:, i] * width + d[0]
    seed_y = function(seed_x)
    procedure.set_seed(seed_x, seed_y)
    
    # Format results folder
    results_folder = RESULTS_FOLDER.format(input_dimensionality, input_width_decay, input_stdev_configs, input_survival_rates)

    # Run experiment
    experiment = hds.OptimisationExperiment(procedure, results_folder)
    experiment.run(function, finish_line=int(MAX_N_SAMPLES))
