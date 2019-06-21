from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

from .utils import get_time


class TestFunction(metaclass=ABCMeta):
    def __init__(self):
        self.is_differentiable = False
        self.ranges = []

    def __call__(self, x, derivative=False):
        # Check if testfunction is fully configured
        self.check_configuration()
        # Convert input information to numpy array
        x = self.to_numpy_array(x)
        # Check if dimensionality and ranges of the input are correct
        self.check_dimensionality(x.shape)
        self.check_ranges(x)
        # Return requested result(s)
        if not derivative:
            return self.evaluate(x)
        else:
            # Check if function differentiability is compatible with request
            self.check_differentiability(derivative)
            # Return derivative if no error was raised by previous line
            return self.derivative(x)

    def check_configuration(self):
        # Check if is_differentiable is known
        if not hasattr(self, "is_differentiable"):
            raise Exception("The testfunction has unknown differentiability.")
        # Check if ranges are known
        if not hasattr(self, "ranges"):
            raise Exception("Testfunction has unknown ranges.")
        # Convert ranges to array if not already an array
        # This is useful for the check_dimensionality and check_ranges methods
        if not isinstance(self.ranges, np.ndarray):
            self.ranges = np.array(self.ranges)

    def check_dimensionality(self, shape):
        dim = self.ranges.shape[0]
        dim_data = shape[1]
        if dim_data != dim:
            raise Exception("Provided data has dimensionality {}, but {} was expected".format(dim_data, dim))
    
    def check_ranges(self, x):
        # Transform data
        d = x - self.ranges[:,0]
        d = d / (self.ranges[:,1] - self.ranges[:,0])
        # Check if any entry smaller than 0 exists
        if np.any(d < 0.0) or np.any(d > 1.0):
            raise Exception("Data does not fall within expected ranges: {}".format(self.ranges))
    
    def check_differentiability(self, derivative):
        if derivative and not self.is_differentiable:
            raise Exception("Derivative could not be calculated: function is not differentiable.")

    def to_numpy_array(self, x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            return np.array(x)
        if isinstance(x, pd.DataFrame):
            return x.values()

    @abstractmethod
    def evaluate(self, x):
        pass
    
    @abstractmethod
    def derivative(self, x):
        pass


class FunctionCallCounter:
    def __init__(self, function):
        self.function = function
        self.counter_time = []
        self.counter_derivatives = []
    
    def __call__(self, x, derivative):
        # Perform function call
        t_start = get_time()
        value = self.function(x, derivative)
        t_end = get_time()

        # Log count and dt
        self.counter_time.append( t_end - t_start )
        self.counter_derivatives.append(1*bool(derivative))
        return value

    def evaluate(self, x):
        return self(x, False)
    
    def derivative(self, x):
        return self(x, True)
    
    def reset(self):
        self.evaluations = [0, 0]
        self.time = [[], []]


class NoDerivativeError(Exception):
    pass


class Sphere(TestFunction):
    def __init__(self, dimensionality=3):
        self.is_differentiable = True
        self.ranges = [[-np.inf, np.inf]]*dimensionality
        self.global_minimum = [0]*dimensionality

    def evaluate(self, x):
        return np.sum(np.power(x, 2), axis=1)
    
    def derivative(self, x):
        return 2*x

class Ackley(TestFunction):
    def __init__(self):
        self.is_differentiable = False
        self.ranges = [[-5, 5]]*2
        self.global_minimum = [0,0]
    
    def evaluate(self, x):
        return -20*np.exp(-0.2*np.sqrt(0.5*(x[:,0]**2, x[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[:,0]) + np.cos(2*np.pi*x[:,1]))) + np.exp(1) + 20
        