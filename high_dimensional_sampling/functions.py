from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

from .utils import get_time


class TestFunction(metaclass=ABCMeta):
    def __init__(self):
        if not hasattr(self, 'is_differentiable'):
            self.is_differentiable = False
        if not hasattr(self, 'ranges'):
            self.ranges = []
            raise Exception("TestFunction should define ranges.")
        self.counter_time = []
        self.counter_derivatives = []

    def __call__(self, x, derivative=False):
        # Check if testfunction is fully configured
        self.check_configuration()
        # Convert input information to numpy array
        x = self.to_numpy_array(x)
        # Check if dimensionality and ranges of the input are correct
        self.check_dimensionality(x.shape)
        self.check_ranges(x)
        # Check if function differentiability is compatible with request
        self.check_differentiability(derivative)
        # Start time for function call
        t_start = get_time()
        # Return requested result(s)
        if not derivative:
            value = self.evaluate(x)
        else:
            # Return derivative if no error was raised by previous line
            value = self.derivative(x)
        # Store call and dt
        self.counter_time.append(get_time() - t_start)
        self.counter_derivatives.append(1*bool(derivative))
        # Return value
        return value

    def reset(self):
        self.evaluations = [0, 0]
        self.time = [[], []]

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


class NoDerivativeError(Exception):
    pass


class Sphere(TestFunction):
    def __init__(self, dimensionality=3):
        self.is_differentiable = True
        self.ranges = []
        for _ in range(dimensionality):
            self.ranges.append([-np.inf, np.inf])
        super(Sphere, self).__init__()

    def evaluate(self, x):
        return np.sum(np.power(x, 2), axis=1)
    
    def derivative(self, x):
        return 2*x

class Ackley(TestFunction):
    def __init__(self):
        self.is_differentiable = False
        self.ranges = [[-5, 5], [-5, 5]]
        super(Ackley, self).__init__()
    
    def evaluate(self, x):
        return -20*np.exp(-0.2*np.sqrt(0.5*(x[:,0]**2, x[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[:,0]) + np.cos(2*np.pi*x[:,1]))) + np.exp(1) + 20


class Easom(TestFunction):
    def __init__(self):
        self.is_differentiable = False
        self.ranges = [[-100, 100], [-100, 100]]
        super(Easom, self).__init__()
    
    def evaluate(self, x):
        return -1*np.cos(x[:,0])*np.cos(x[:,1])*np.exp(-1*(np.power(x[:,0]-np.pi, 2) + np.power(x[:,1]-np.pi, 2)))