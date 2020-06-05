import sys
import high_dimensional_sampling as hds
from high_dimensional_sampling import functions as func

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


class ParticleFilter(hds.Procedure):
    def __init__(self,
                 seed_size=1000,
                 seed_distributions=None,
                 ranges=None,
                 hard_ranges=False,
                 iteration_size=1000,
                 selector_function=None,
                 width=2,
                 width_scheduler=None,
                 width_decay=0.95,
                 gaussian_constructor=None,
                 inf_replace=1e9):
        """Implementation of the Particle Filter algorithm

        The documentation in the code can be found at the project wiki on
        Github:
        https://github.com/DarkMachines/high-dimensional-sampling/wiki/Particle-Filter.

        Args:
            seed_size: Number of points (int) to sample at the start of the
                procedure. Default: 1000.
            seed_distributions: Distributions from which the seed will be
                sampled. Should be provided as a list of strings, either
                "linear" or "log". The default is to set all distributions to
                linear (i.e. uniform) within the specified `ranges`.
            ranges: Ranges within which the sampling should take place. See
                also the `hard_ranges` property. By default this is set to the
                rangesof the test function itself.
            hard_ranges: Boolean. If set to `True`, the ranges in the `ranges`
                property hold for all sampling iterations. If it is `False`,
                the ranges only hold for the initial seed sampling. Default:
                False.
            iteration_size: Number of samples to add in each iteration.
                Default: 1000.
            selector_function: Function to sample points from previous samples.
                Default: `self.selector_deterministic_linear`.
            width: Start width parameter for the gaussians. Default: 2.
            width_scheduler: Function that schedules the changes in width over
                the iterations. Default: `self.width_schedule_exponential`.
            width_decay: Decay parameter for the width. Used by the width
                scheduler. Default: 0.95.
            gaussian_constructor: Function that returns the stdevs of the
                gaussians to use. Default: `gaussian_constructor_linear`.
            inf_replace: Number to replace infinities in ranges with.
                Default: 1e9.
        """
        # Iteration counter
        self.iteration = 0
        # Samples from the previous iteration
        self.previous_samples = None

        # Number of samples in initial seed sampling
        self.seed_size = int(seed_size)
        # Distributions of initial seed sampling
        self.seed_distributions = seed_distributions

        # Number of samples in each iteration
        self.iteration_size = int(iteration_size)
        # Function to sample points from previous samples
        self.selector_function = selector_function
        # Start width parameter for the gaussians
        self.initial_width = float(width)
        self.width = float(width)
        # Function that schedules the changes in width
        self.width_scheduler = width_scheduler
        # Decay parameter for the width
        self.width_decay = width_decay
        # Function that returns the stdevs of the gaussians to use
        self.gaussian_constructor = gaussian_constructor

        # Ranges within which the sampling should take place. See also the
        # `hard_ranges` property.
        self.ranges = ranges
        # If the `hard_ranges` is `True`, the ranges in the `ranges` property
        # hold for all sampling iterations. If it is `False`, the ranges only
        # hold for the initial seed sampling.
        self.hard_ranges = bool(hard_ranges)

        # Miscelaneous
        # Number to replace infinities in ranges with
        self.inf_replace = inf_replace
        # List of parameters to store when the procedure is used in an
        # experiment
        self.store_parameters = [
            'inf_replace', 'seed_size', 'seed_distributions', 'iteration_size',
            'initial_width', 'width', 'width_decay', 'ranges', 'hard_ranges'
        ]

    def __call__(self, function):
        self.check_testfunction(function)
        if self.previous_samples is None:
            # Sample seed using user defined seed distributions
            x, y = self.sample_seed(function)
        else:
            # Sample new iteration with gaussian kernel
            x, y = self.sample_iteration(function)
        self.iteration += 1
        self.previous_samples = (x, y)
        return (x, y)

    def check_testfunction(self, function):
        """ Checks if the particle filter can work with the provided function.

        """

        # Check if the particle filter was fully configured and configure
        # missing values using the properties of the provided function.
        self.configure_missing_values(function)
        # Check if the configuration can be run on the provided test function.
        # Do the ranges match the dimensionality of the function?
        if len(self.ranges) != function.get_dimensionality():
            return False
        # Do the seed distributions match this same dimensionality?
        if len(self.seed_distributions) != function.get_dimensionality():
            return False
        # Are the seed distributions all "linear", "uniform" or "log"?
        for d in set(self.seed_distributions):
            if d not in ['linear', 'log', 'uniform']:
                return False
        return True

    def is_finished(self):
        return False

    def reset(self):
        self.iteration = 0
        self.previous_samples = None
        self.seed_distributions = None
        self.width = self.initial_width
        self.ranges = None

    """ ==================================== Particle filter iterations === """

    def sample_iteration(self, function):
        # Select points to use as seed for gaussian
        ind, samples, values = self.selector_function(self, self.iteration_size,
                                                      self.previous_samples[0],
                                                      self.previous_samples[1])
        selected = samples[ind]
        values = values[ind]
        # Determine sigmas for gaussians
        stdevs = self.gaussian_constructor(self, selected, values)
        # Sample from gaussians
        x = np.zeros((self.iteration_size, selected.shape[1]))
        for i, r in enumerate(self.ranges):
            if self.hard_ranges:
                x[:, i] = self._sample_iteration_hard(selected[:, i],
                                                      stdevs[:, i], r[0], r[1])
            else:
                x[:, i] = self._sample_iteration_soft(selected[:, i],
                                                      stdevs[:, i])
        y = function(x)
        return (x, y)

    def _sample_iteration_soft(self, means, stdevs):
        return np.random.normal(means, stdevs)

    def _sample_iteration_hard(self, means, stdevs, minimum, maximum):
        x = np.zeros(means.shape)
        for i, (mean, stdev) in enumerate(zip(means, stdevs)):
            x[i] = stats.truncnorm.rvs((minimum - mean) / stdev,
                                       (maximum - mean) / stdev,
                                       loc=mean,
                                       scale=stdev)
        return x

    def determine_gaussian_width(self):
        if isinstance(self.width_scheduler, float):
            return width_schedule_exponential(self, self.width_decay)
        return self.width_scheduler(self, self.width_decay)

    """ ================================================== Seed methods === """

    def set_seed(self, seed, function):
        self.iteration += 1
        self.previous_samples = (seed, function(seed))

    def sample_seed(self, function):
        x = np.zeros((self.seed_size, function.get_dimensionality()))
        # Loop over all dimensions and fill the samples array with values
        print("ranges: {}".format(self.ranges))
        for i, (r, dist) in enumerate(zip(self.ranges,
                                          self.seed_distributions)):
            if dist == 'log':
                x[:, i] = self._sample_log(self.seed_size, r[0], r[1])
            else:
                x[:, i] = self._sample_uniform(self.seed_size, r[0], r[1])
        # Evaluate function at samples
        y = function(x)
        return (x, y)

    def _sample_uniform(self, n, minimum, maximum):
        width = maximum - minimum
        offset = minimum
        return np.random.rand(n) * width + offset

    def _sample_log(self, n, minimum, maximum):
        if minimum >= 0:
            if minimum == 0:
                minimum = sys.float_info.min
            # All samples in positive range
            x = self._sample_uniform(n, np.log10(minimum), np.log10(maximum))
            return np.power(10, x)
        elif maximum <= 0:
            if maximum == 0:
                maximum = -1 * sys.float_info.min
            # All samples in negative range
            x = self._sample_uniform(n, np.log10(-minimum), np.log10(-maximum))
            return -1 * np.power(10, x)
        else:
            # Samples in both negative and positive range
            p = np.abs(minimum) / (maximum - minimum)
            n_negative = np.random.binomial(n, p)
            x = np.zeros(n)
            x[:n_negative] = self._sample_log(n_negative, minimum,
                                              -1 * sys.float_info.min)
            x[n_negative:] = self._sample_log(n - n_negative,
                                              sys.float_info.min, maximum)
            return x

    """ ========================================= Configuration methods === """

    def configure_missing_values(self, function):
        # Configure ranges using the test function's ranges
        if self.ranges is None:
            self.ranges = function.get_ranges()

        # Configure seed distributions (default to all linear)
        if self.seed_distributions is None:
            self.seed_distributions = ['linear'
                                       ] * function.get_dimensionality()

        # Configure the width scheduler to default if none provided
        if not hasattr(self.width_scheduler, '__call__'):
            self.width_scheduler = width_schedule_exponential
        # Configure default weiging function
        if not hasattr(self.selector_function, '__call__'):
            self.selector_function = selector_stochastic_linear
        # Configure the default gaussian stdev constructor
        if not hasattr(self.gaussian_constructor, '__call__'):
            self.gaussian_constructor = gaussian_constructor_linear
        # Replace infinities with the infinity replacer property
        if len(self.ranges[np.abs(self.ranges) == np.inf]) != 0:
            print("""Warning: The particle filter replaced an infinity /
                multiple infinities in the sampling ranges with
                {}.""".format(self.inf_replace))
        self.ranges[self.ranges == np.inf] = self.inf_replace
        self.ranges[self.ranges == -np.inf] = -self.inf_replace


""" ===================================== Gaussian construction methods === """


def gaussian_constructor_linear(algorithm, samples, values):
    width = algorithm.determine_gaussian_width()
    ranges = algorithm.ranges[:, 1] - algorithm.ranges[:, 0]
    ranges[ranges == 0] = 1.0
    return np.abs(width * ranges * np.ones(samples.shape))


def gaussian_constructor_log(algorithm, samples, values):
    width = algorithm.determine_gaussian_width()
    ranges = algorithm.ranges[:, 1] - algorithm.ranges[:, 0]
    ranges[ranges == 0] = 1.0
    return np.abs(width * ranges * samples)


""" =========================================== Width parameter methods === """


def width_schedule_exponential(algorithm, alpha):
    algorithm.width = algorithm.width * alpha
    return algorithm.width


def width_schedule_exponential_10stepped(algorithm, alpha):
    if algorithm.iteration % 10 == 0:
        algorithm.width = algorithm.width * alpha
    return algorithm.width


""" =========================================== Sample selector methods === """


def selector_deterministic_linear(algorithm, n, samples, values):
    # Calculate probabilities for samples
    z = values - np.amin(values)
    if len(np.unique(z)) != 1:
        z = 1 - (z / np.amax(z))
        probabilities = z / np.sum(z)
    else:
        probabilities = np.ones(len(z))/len(z)

    # Sort samples based on probability, from low to high
    sortind = np.argsort(probabilities)[::-1]
    z = z[sortind]
    samples = samples[sortind]
    values = values[sortind]
    probabilities = probabilities[sortind]
    
    # Determine samples per point
    samples_per_ind = np.ceil(probabilities*n)

    # Correct for rounding errors
    at_least_one = 1.0*(samples_per_ind > 0)
    for i in range(len(at_least_one)):
        potentially_sampled = np.sum(samples_per_ind) - np.sum(at_least_one)
        if potentially_sampled < n:
            at_least_one[i] = 0
    samples_per_ind = samples_per_ind - at_least_one

    # Create indices and return values
    indices = []
    for i in range(len(samples_per_ind)):
        indices.extend([i]*int(samples_per_ind[i]))
    return (indices, samples, values)


def selector_stochastic_uniform(algorithm, n, samples, values):
    indices = np.random.choice(len(samples), n)
    return (indices, samples, values)


def selector_stochastic_linear(algorithm, n, samples, values):
    z = values - np.amin(values)
    if len(np.unique(z)) != 1:
        z = 1 - (z / np.amax(z))
        probabilities = z / np.sum(z)
    else:
        probabilities = np.ones(len(z))/len(z)
    indices = np.random.choice(len(samples), n, p=probabilities.flatten())
    return (indices, samples, values)


def selector_stochastic_softmax(algorithm, n, samples, values):
    z = values - np.amin(values)
    probabilities = np.exp(-z) / np.sum(np.exp(-z))
    indices = np.random.choice(len(samples), n, p=probabilities)
    return (indices, samples, values)


class ExampleFunction(func.TestFunction):
    def __init__(self):
        self.ranges = [[-10, 10], [-10, 10], [-10, 10]]
        super(ExampleFunction, self).__init__()

    def _evaluate(self, x):
        return np.sum(x**2, axis=1)

    def _derivative(self, x):
        raise func.NoDerivativeError()


if __name__ == "__main__":
    # Define ingredients
    algorithm = ParticleFilter(seed_size=100,
                               iteration_size=100,
                               width_decay=0.8,
                               hard_ranges=True)
    function = ExampleFunction()
    # Run algorithm on function
    algorithm.check_testfunction(function)
    x0, y0 = algorithm(function)
    for _ in range(25):
        _, _ = algorithm(function)
    xn, yn = algorithm(function)

    plt.hist(x0[:, 0], 100, normed=1, histtype='step', label='seed')
    plt.hist(xn[:, 0], 100, normed=1, histtype='step', label='iteration 16')
    plt.legend()
    plt.show()

    algorithm = ParticleFilter(seed_size=100,
                               iteration_size=100,
                               width_decay=0.8,
                               hard_ranges=True,
                               width_scheduler=width_schedule_exponential)
