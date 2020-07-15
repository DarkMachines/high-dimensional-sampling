import sys
import high_dimensional_sampling as hds
from high_dimensional_sampling import functions as func
import particlefilter as pf

import numpy as np
import matplotlib.pyplot as plt


class GaussianParticleFilter(hds.Procedure):
    def __init__(self,
                 seed_size=100,
                 iteration_size=100,
                 boundaries=None,
                 initial_width=2,
                 wc_decay_rate=0.95,
                 wc_apply_every_n_iterations=1,
                 sc_min_stdev=0.0,
                 sc_max_stdev=np.inf,
                 sc_scales_with_boundary=True,
                 sc_logarithmic=False,
                 kc_survival_rate=0.2,
                 kc_cut_to_iteration_size=False,
                 max_resample_attempts=100,
                 inf_replace=1e12):
        
        # Store properties
        self.iteration = 0
        self.seed_size = seed_size
        self.wc_decay_rate = wc_decay_rate
        self.wc_apply_every_n_iterations = wc_apply_every_n_iterations
        self.sc_min_stdev = sc_min_stdev
        self.sc_max_stdev = sc_max_stdev
        self.sc_scales_with_boundary = sc_scales_with_boundary
        self.sc_logarithmic = sc_logarithmic
        self.kc_survival_rate = kc_survival_rate
        self.kc_cut_to_iteration_size = kc_cut_to_iteration_size

        # Create core object
        self.pf = pf.ParticleFilter(
            function=lambda x: np.sum(x, axis=1),
            iteration_size=iteration_size,
            boundaries=boundaries,
            initial_width=initial_width,
            width_controller=pf.get_width_controller(
                self.wc_decay_rate,
                self.wc_apply_every_n_iterations
            ),
            stdev_controller=pf.get_stdev_controller(
                self.sc_min_stdev,
                self.sc_max_stdev,
                self.sc_scales_with_boundary,
                self.sc_logarithmic,
                inf_replace
            ),
            kill_controller=pf.get_kill_controller(
                self.kc_survival_rate,
                self.kc_cut_to_iteration_size
            ),
            max_resample_attempts=max_resample_attempts,
            inf_replace=inf_replace)

        self.store_parameters = [
            'seed_size',
            'iteration_size',
            'initial_width',
            'wc_decay_rate',
            'wc_apply_every_n_iterations',
            'sc_min_stdev',
            'sc_max_stdev',
            'sc_scales_with_boundary',
            'sc_logarithmic',
            'kc_survival_rate',
            'kc_cut_to_iteration_size',
            'boundaries',
            'inf_replace'
        ]
    def __len__(self):
        return len(self)
    
    def __getattribute__(self, name):
        if name in ['iteration_size', 'initial_width', 'inf_replace',
            'boundaries']:
            v = getattr(self.pf, name)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if name == 'inf_replace':
                v = float(v)
            return v
        return object.__getattribute__(self, name)

    def add_callback(self, name, function=None):
        self.pf.add_callback(name, function)
    
    def set_seed(self, x, y):
        self.pf.set_seed(x, y)

    def __call__(self, function):
        # Insert function in particle filter
        func = lambda x: function(x)
        func = self.pf.validate_function(func)
        self.pf.function = func
        # Replace boundaries 
        if not np.array_equal(self.pf.boundaries, function.ranges):
            self.pf.boundaries = np.array(function.ranges)
        # Sample!
        if self.pf.population is None or len(self.pf.population) == 0:
            # Sample seed uniformly
            self.pf.sample_seed(self.seed_size)
        else:
            # Sample new iteration with gaussian kernel
            self.pf.run_iteration()
        # Return latest iteration samples
        x, y = self.pf.population.get_data_by_origin(self.iteration)
        print("Iteration {}".format(self.iteration))
        for sp in self.store_parameters:
            print(sp, getattr(self, sp))
        print("\n\n")
        self.iteration += 1
        return (x, y)
    
    def check_testfunction(self, function):
        return True

    def is_finished(self):
        return False

    def reset(self):
        self.pf.reset()


class ExampleFunction(func.TestFunction):
    def __init__(self):
        self.ranges = [[-10, 10], [-10, 10], [-10, 10]]
        super(ExampleFunction, self).__init__()

    def _evaluate(self, x):
        x = x - np.array([-3, 4, 5])
        return np.sum(x**2, axis=1)

    def _derivative(self, x):
        raise func.NoDerivativeError()


if __name__ == "__main__":
    # Define ingredients
    algorithm = GaussianParticleFilter(seed_size=100,
                                       iteration_size=100,
                                       initial_width=0.2,
                                       wc_decay_rate=0.8)
    function = ExampleFunction()
    # Run algorithm on function
    algorithm.check_testfunction(function)
    x0, y0 = algorithm(function)
    for _ in range(25):
        _, _ = algorithm(function)
    xn, yn = algorithm(function)

    bins = np.linspace(-10, 10, 25)
    plt.hist(x0[:, 0], bins, density=1, histtype='step', label='seed')
    plt.hist(xn[:, 0], bins, density=1, histtype='step', label='iteration 24')
    plt.legend()
    plt.show()

