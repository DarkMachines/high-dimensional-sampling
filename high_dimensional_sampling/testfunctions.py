from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class TestFunction(metaclass=ABCMeta):
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
        # Return requested result(s)
        if not derivative:
            return self.evaluate(x)
        else:
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
        if derivative:
            if not self.is_differentiable:
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


class FunctionWrapper:
    def __init__(self, function):
        self.function = function
        self.evaluations = [0, 0]
        self.time = [[], []]

    def evaluate(self, x):
        raise NotImplementedError
    
    def derivative(self, x):
        raise NotImplementedError


class Sine(TestFunction):
    def __init__(self):
        self.is_differentiable = True
        self.ranges = [[-np.inf, np.inf]]

    def evaluate(self, x):
        return np.sin(x)
    
    def derivative(self, x):
        return np.cos(x)


"""
class MexicanHat(TestFunction):
    def __init__(self, fourth_order=2, second_order=-2):
        self.is_differentiable = True
        self.dimensionality = 2
        self.ranges = [[-10,10], [-10,10]]
        self.fourth_order = fourth_order
        self.second_order = second_order
    
    def evaluate(self, x):
        r = np.linalg.norm(x, axis=1)
        return self.fourth_order * np.power(r, 4) + self.second_order * np.power(r, 2)
    
    def derivative(self, x):
        r = np.linalg.norm(x, axis=1)
        return 4 * self.fourth_order * np.power(r, 3) + 2 * self.second_order * r
"""
