from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from .utils import get_time


class TestFunction(ABC):
    """
    Abstract base class for test functions

    This class forms the basis for all testfunctions implemented in this
    module. It automatically tracks the number of evaluations of the function
    and does dimensionality checks on the input data.

    As this is an abstract class, instances of this class cannot be created or
    used.

    Raises:
        Exception: Testfunction should define ranges.
    """

    def __init__(self):
        if not hasattr(self, 'ranges'):
            self.ranges = []
            raise Exception("TestFunction should define ranges.")
        self.counter = []

    def __call__(self, x, derivative=False):
        """
        Request an evaluation of the (derivative of the) testfunction.

        After having performed dimensionality and ranges checks on the
        provided parameters x, the testfunction in the derived class is queried
        for its value or its derivative (depending on the request by the user).
        In this proces the query time is stored for later logging.

        Args:
            x: numpy.ndarray of shape (nDatapoints, nVariables) containing the
                datapoints to query to the testfunction
            derivative: If this boolean is False (default), the testfunction
                will be queried for its value. If it is True, the derivative
                of the function is queried (if existing).

        Returns:
            A numpy.ndarray of shape (nDatapoints, ?) containing the requested
            values.
        """
        # Check if testfunction is fully configured
        self.check_configuration()
        # Convert input information to numpy array
        x = self.to_numpy_array(x)
        # Check if dimensionality and ranges of the input are correct
        self.check_dimensionality(x.shape)
        self.check_ranges(x)
        # Start time for function call
        t_start = get_time()
        # Return requested result(s)
        if not derivative:
            value = self._evaluate(x)
        else:
            # Return derivative if no error was raised by previous line
            value = self._derivative(x)
        # Store call and dt
        self.counter.append([len(x), get_time() - t_start, bool(derivative)])
        # Return value
        return value

    def reset(self):
        """
        Resets the internal counter of queries
        """
        self.counter = []

    def check_configuration(self):
        """
        Checks if the testfunction is correctly configured

        As this class is an abstract base class, we want to make sure that the
        derived class is configured correctly. This method performs sanity
        checks to make us sure of this. It checks if the ranges to sample are
        specified. If either of these checks fails, an Exception is raised.

        Raises:
            Exception: The testfunction has unknown differentiability.
            Exception: Testfunction has unknown ranges.
        """
        # Check if ranges are known
        if not hasattr(self, "ranges"):
            raise Exception("Testfunction has unknown ranges.")
        # Convert ranges to array if not already an array
        # This is useful for the check_dimensionality and check_ranges methods
        if not isinstance(self.ranges, np.ndarray):
            self.ranges = np.array(self.ranges)

    def check_dimensionality(self, shape):
        """
        Checks if the dimensionality of the input data is conform expected by
        the defined testfunction. If not, an Exception is raised.

        Args:
            shape: Tuple defining the shape of the input data.

        Raises:
            Exception: Provided data has dimensionality ?, but ? was expected.
        """
        dim = self.ranges.shape[0]
        dim_data = shape[1]
        if dim_data != dim:
            raise Exception(
                "Provided data has dimensionality {}, but {} was expected".
                format(dim_data, dim))

    def check_ranges(self, x):
        """
        Checks if the input data lies within the ranges expected by the
        defined testfunction. If not, an Exception is raised.

        Args:
            x: numpy.ndarray of shape (nDatapoints, nVariables) containing the
                data to query the testfunction for.

        Raises:
            Exception: Data does not fall withing the expected ranges: ?.
        """
        # Transform data
        d = x - self.ranges[:, 0]
        d = d / (self.ranges[:, 1] - self.ranges[:, 0])
        # Check if any entry smaller than 0 exists
        if np.any(d < 0.0) or np.any(d > 1.0):
            raise Exception(
                "Data does not fall within expected ranges: {}.".format(
                    self.ranges))

    def to_numpy_array(self, x):
        """
        Converts data to an numpy array

        Converts variables of type list and pandas.DataFrame to a numpy.ndarray
        and returns these. Variables of type numpy.ndarray are left untouched.

        Args:
            x: Data of type numpy.ndarray, list or pandas.DataFrame.

        Returns:
            Data converted to numpy.ndarray.

        Raises:
            Exception: Testfunctions don't accept ? as input: only numpy
                arrays, list and pandas dataframes are allowed.
        """
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            return np.array(x)
        if isinstance(x, pd.DataFrame):
            return x.values()
        raise Exception(
            """"Testfunctions don't accept {} as input: only numpy arrays, 
            lists and pandas dataframes are allowed.""".format(
                type(x).__name__))

    def construct_ranges(self, dimensionality, min, max):
        """
        Constructs the application range of the test function for a dynamic
        dimensionality

        Args:
            dimensionality: Number of dimensions
            min: Minimum value for all dimensions
            max: Maximum value for all dimensions

        Returns:
            A list containing the minimum and maximum values for all
            dimensions.
        """
        ranges = []
        for _ in range(dimensionality):
            ranges.append([min, max])
        return ranges

    @abstractmethod
    def _evaluate(self, x):
        """
        Queries the testfunction for a function evaluation.

        This method should be implemented by any testfunction derived from
        this abstract base class.

        Args:
            x: Data as a numpy.ndarray of shape (nDatapoints, nVariables) for
                which the testfunction should be evaluated.

        Returns:
            Values returned by the function evaluation as numpy.ndarray of
            shape (nDatapoints, ?).
        """
        pass

    def _derivative(self, x):
        """
        Queries the testfunction for its derivative at the provided point(s).

        This method should be implemented by any testfunction derived from
        this abstract base class *if* a derivative is known.

        Args:
            x: Data as a numpy.ndarray of shape (nDatapoints, nVariables) for
                which the gradient of the testfunction should be returned.

        Returns:
            Gradient of the testfunction at the provided location as
            numpy.ndarray of shape (nDatapoints, ?).

        Raises:
            NoDerivativeError: No derivative is known for this testfunction.
        """
        raise NoDerivativeError()


class NoDerivativeError(Exception):
    """
    Exception indicating no derivative is known to queried testfunction
    """
    pass


class Rastrigin(TestFunction):
    """
    Testfunction as defined by 
    https://en.wikipedia.org/wiki/Rastrigin_function
    """

    def __init__(self, dimensionality=2):
        self.ranges = self.construct_ranges(dimensionality, -5.12, 5.12)
        self.a = 10
        super(Rastrigin, self).__init__()

    def _evaluate(self, x):
        n = len(self.ranges)
        y = self.a * n
        for i in range(n):
            y += np.power(x[:, i], 2) - self.a * np.cos(2 * np.pi * x[:, i])
        return y

    def _derivative(self, x):
        return 2 * x + 2 * np.pi * self.a * np.sin(2 * np.pi * x)


class Rosenbrock(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Rosenbrock_function

    No derivative implemented
    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rosenbrock function has to
                            be >=2.""")
        self.ranges = self.construct_ranges(dimensionality, -np.inf, np.inf)
        super(Rosenbrock, self).__init__()

    def _evaluate(self, x):
        n = len(self.ranges)
        y = 0
        for i in range(n - 1):
            y += (100 * np.power(x[:, i + 1] - np.power(x[:, i], 2), 2) +
                  np.power(1 - x[:, i], 2))
        return y


class Beale(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-4.5, 4.5], [-4.5, 4.5]]
        super(Beale, self).__init__()

    def _evaluate(self, x):
        return (np.power(1.5 - x[:, 0] + x[:, 0] * x[:, 1], 2) +
                np.power(2.25 - x[:, 0] + x[:, 0] * np.power(x[:, 1], 2), 2) +
                np.power(2.625 - x[:, 0] + x[:, 0] * np.power(x[:, 1], 3), 2))


class Booth(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-10, 10], [-10, 10]]
        super(Booth, self).__init__()

    def _evaluate(self, x):
        return np.power(x[:, 0] + 2 * x[:, 1] - 7, 2) + np.power(
            2 * x[:, 0] + x[:, 1] - 5, 2)


class BukinNmbr6(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-15, -5], [-3, 3]]
        super(BukinNmbr6, self).__init__()

    def _evaluate(self, x):
        return 100 * np.sqrt(
            np.abs(x[:, 1] - 0.01 * np.power(x[:, 0], 2)) +
            0.01 * np.abs(x[:, 0] + 10))


class Matyas(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-10, 10], [-10, 10]]
        super(Matyas, self).__init__()

    def _evaluate(self, x):
        return 0.26 * (np.power(x[:, 0], 2) +
                       np.power(x[:, 1], 2)) - 0.48 * x[:, 0] * x[:, 1]


class LeviNmbr13(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-10, 10], [-10, 10]]
        super(LeviNmbr13, self).__init__()

    def _evaluate(self, x):
        return (np.power(np.sin(3 * np.pi * x[:, 0]), 2) +
                np.power(x[:, 0] - 1, 2) *
                (1 + np.power(np.sin(3 * np.pi * x[:, 1]), 2)) +
                np.power(x[:, 1] - 1, 2) *
                (1 + np.power(np.sin(2 * np.pi * x[:, 1]), 2)))


class Himmelblau(TestFunction):
    """
    Testfunction as defined by 
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-5, 5], [-5, 5]]
        super(Himmelblau, self).__init__()

    def _evaluate(self, x):
        return (np.power(np.power(x[:, 0], 2) + x[:, 1] - 11, 2) +
                np.power(x[:, 0] + np.power(x[:, 1], 2) - 7, 2))


class ThreeHumpCamel(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative implemented
    """

    def __init__(self):
        self.ranges = [[-5, 5], [-5, 5]]
        super(ThreeHumpCamel, self).__init__()

    def _evaluate(self, x):
        return (2.0 * np.power(x[:, 0], 2) - 1.05 * np.power(x[:, 0], 4) +
                np.power(x[:, 0], 6) / 6.0 + x[:, 0] * x[:, 1] +
                np.power(x[:, 1], 2))


class Sphere(TestFunction):
    """
    Testfunction that returns the squared euclidean distance on evaluation.

    Testfunction following the formula:

        y = sum_{i=1}^{N} x_i^2

    The derivative of this function is implemented as

        y' = 2*x
    """

    def __init__(self, dimensionality=3):
        self.ranges = self.construct_ranges(dimensionality, -np.inf, np.inf)
        super(Sphere, self).__init__()

    def _evaluate(self, x):
        return np.sum(np.power(x, 2), axis=1)

    def _derivative(self, x):
        return 2 * x


class Ackley(TestFunction):
    """
    Ackley function as defined by
    https://en.wikipedia.org/wiki/Ackley_function.

    No derivative has been implemented.
    """

    def __init__(self):
        self.ranges = [[-5, 5], [-5, 5]]
        super(Ackley, self).__init__()

    def _evaluate(self, x):
        a = -20 * np.exp(
            -0.2 * np.sqrt(0.5 *
                           (np.power(x[:, 0], 2) + np.power(x[:, 1], 2))))
        f = np.cos(2 * np.pi * x[:, 0])
        g = np.cos(2 * np.pi * x[:, 1])
        b = -np.exp(0.5 * (f + g))
        return a + b + np.exp(1) + 20


class Easom(TestFunction):
    """
    Easom function as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    No derivative has been implemented.
    """

    def __init__(self):
        self.ranges = [[-100, 100], [-100, 100]]
        super(Easom, self).__init__()

    def _evaluate(self, x):
        return (-1 * np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.exp(
            -1 *
            (np.power(x[:, 0] - np.pi, 2) + np.power(x[:, 1] - np.pi, 2))))
