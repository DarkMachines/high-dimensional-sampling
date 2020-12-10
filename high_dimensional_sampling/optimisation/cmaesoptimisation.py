""" Implementation of the CMA-ES optimisation. Documentation for the cma
package is at https://pypi.org/project/cma/. The package can be installed with
pip. The options of the package are currently set with a dictionary, see the
CMAOptions package for details.

It is implicitly assumed that the parameters to be optimized in the function
have been scaled properly to be of the order of 1. """
import high_dimensional_sampling as hds
import numpy as np

try:
    import cma
except ImportError:
    pass


class CMAOptimisation(hds.Procedure):
    def __init__(self, options={}):
        """
        Initializes the algorithm.  The options dictionary is passed to the
        CMAOptions object
        """
        try:
            cma
        except NameError:
            raise ImportError(
                "The `cma` package is not installed. See the wiki on our "
                "GitHub project for installation instructions.")

        self.store_parameters = []
        self.opts = cma.CMAOptions()
        self.opts.init(options)
        self.es = None
        self.reset()

    def __call__(self, function):
        """ Get ranges of the test function. The 0.001 moves the minima 0.001
        up and the maxima 0.001 down, in order to make use the sampling is not
        by accident moving outside of the test function range. """
        ranges = function.get_ranges(1e-11)

        if self.es is None:
            # Set up the sampler.  Use the center of the range as initial
            # position and a third of the range as the initial sigma to cover
            # nearly the entire parameter space initially.
            # Unbounded parameters are an issue, sample those from -3 to 3
            # This implicitly assumes the best fit value is close to 0 for
            # the unbounded parameters
            r = np.array(ranges)
            r[:, 0][np.isinf(r[:, 0])] = -3
            r[:, 1][np.isinf(r[:, 1])] = 3
            self.__set_scale__(r)
            x0 = np.zeros_like(r[:, 0])
            sigma = 1./3.

            # Set the bounds before initializing the optimizer
            self.opts["bounds"] = [-1, 1]
            self.es = cma.CMAEvolutionStrategy(x0, sigma, self.opts)

        # Get new points
        xprime = np.array(self.es.ask())
        x = self.__scale__(xprime)
        y = function(x)
        self.es.tell(xprime, [val[0] for val in y])

        # Return only the best point of the population
        # THIS IS DISABLED
        # i = np.argmin(y)
        # return (x[i].reshape((1,len(x[0]))), y[i].reshape(-1,1))

        # All points are returned
        return (x, y)

    def __set_scale__(self, ranges):
        # Do linear transformation from range to [-1,1]
        self.scale = (ranges[:, 1]-ranges[:, 0])*0.5
        self.shift = (ranges[:, 1]+ranges[:, 0])/(ranges[:, 1]-ranges[:, 0])

    def __scale__(self, xprime):
        return (xprime+self.shift)*self.scale

    def check_testfunction(self, function):
        return True

    def reset(self):
        self.es = None

    def is_finished(self):
        if (self.es is None):
            return False
        return self.es.stop()
