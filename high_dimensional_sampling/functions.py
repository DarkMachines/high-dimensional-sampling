from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import special, stats
from .utils import get_time


class TestFunction(ABC):
    """
    Abstract base class for test functions

    This class forms the basis for all testfunctions implemented in this
    module. It automatically tracks the number of evaluations of the function
    and does dimensionality checks on the input data.

    As this is an abstract class, instances of this class cannot be created or
    used.

    Properties:
        name: String indicating under which name the TestFunction will be
            logged.

    Raises:
        Exception: Testfunction should define ranges.
    """

    def __init__(self, name=None):
        if not hasattr(self, 'ranges'):
            self.ranges = []
            raise Exception("TestFunction should define ranges.")
        self.counter = []
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

    def __call__(self, x, derivative=False, epsilon=0):
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
            epsilon: leeway parameter that is added to all minima and
                subtracted from all maxima in the .check_ranges method. Default
                is 0.

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
        self.check_ranges(x, epsilon)
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

    def get_ranges(self, epsilon=0.01):
        """
        Get ranges of the test function.

        Returns the ranges for the test function. When numerical precision is a
        problem, a leeway parameter epsilon can be provided, which will be
        added to all minima and subtracted from all maxima.

        Args:
            epsilon: leeway parameter that is added to all minima and
                subtracted from all maxima. Default is 0.001.

        Returns:
            List of minima and maxima for all dimensions in the problem. The
            list has a length equal to the number of dimensions. Each entry in
            this list is a list with two entries: the minimum and the maximum
            for this dimension.
        """
        return np.array([[r[0] + epsilon, r[1] - epsilon]
                         for r in self.ranges])

    def check_ranges(self, x, epsilon=0):
        """
        Checks if the input data lies within the ranges expected by the
        defined testfunction. If not, an Exception is raised.

        Args:
            x: numpy.ndarray of shape (nDatapoints, nVariables) containing the
                data to query the testfunction for.
            epsilon: leeway parameter that is added to all minima and
                subtracted from all maxima. Default is 0.

        Raises:
            Exception: Data does not lie withing the expected ranges: ?.
        """
        ranges = self.get_ranges(epsilon)
        # Transform data
        d = x - ranges[:, 0]
        d = d / (ranges[:, 1] - ranges[:, 0])
        # Check if any entry smaller than 0 exists
        if np.any(d < 0.0) or np.any(d > 1.0):
            raise Exception(
                "Data does not lie within expected ranges: {}.".format(
                    self.ranges.tolist()))

    def count_calls(self, select="all"):
        """
        Counts the number of performed function calls.

        Args:
            select: String indicating which function calls to count. If "all",
                all calls will be counted (default). "normal" makes the
                method only count non-derivate evaluations, whereas
                "derivative" counts the number of derivates evaluated.

        Returns:
            Number of function calls of the selected type.

        Raises:
            Exception: Cannot count function calls of unknown type '?'. Will be
                raised if the select argument is not recognised.
        """
        if select == "all":
            return len(self.counter)
        elif select == "normal":
            n = 0
            for x in self.counter:
                n += 1 - 1 * x[1]
            return round(n)
        elif select == "derivative":
            n = 0
            for x in self.counter:
                n += 1 * x[1]
            return round(n)
        else:
            raise Exception("Cannot count function calls of"
                            "unknown type '{}'".format(select))

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

    def construct_ranges(self, dimensionality, minimum, maximum):
        """
        Constructs the application range of the test function for a dynamic
        dimensionality

        Args:
            dimensionality: Number of dimensions
            minimum: Minimum value for all dimensions
            maximum: Maximum value for all dimensions

        Returns:
            A list containing the minimum and maximum values for all
            dimensions.
        """
        ranges = []
        for _ in range(dimensionality):
            ranges.append([minimum, maximum])
        return ranges

    def get_simple_interface(self):
        """
        Get this function, wrapped in the SimpleFunctionWrapper. This wrapped
        function has a different __call__ interface. See the documentation
        for the wrapper for more information.

        Returns:
            This TestFunction wrapped in a SimpleFunctionWrapper instance.
        """
        return SimpleFunctionWrapper(self)
    
    def is_bounded(self):
        """
        Checks if the ranges of the TestFunction are bounded, i.e. that there
        is no dimension with either np.inf or -np.inf as boundary (or both).

        Returns:
            Boolean indicating if the function is bounded.
        """
        for dim in self.ranges:
            if abs(dim[0]) + abs(dim[1]) == np.inf:
                return False
        return True
    
    def is_differentiable(self):
        """
        Checks if the function is differentiable

        Returns:
            Boolean indicating if function is differentiable.
        """
        x = np.random.rand(self.get_dimensionality())
        ranges = np.array(self.ranges)
        sample = x * (ranges[:,1] - ranges[:,0]) + ranges[:,0]
        sample = sample.reshape((1,-1))
        try:
            _ = self._derivative(sample)
            return True
        except NoDerivativeError:
            return False

    def get_dimensionality(self):
        """
        Returns the dimensionality of the TestFunction, based on the ranges
        defined in the .ranges property.

        Returns:
            Number of dimensions as an integer.
        """
        return len(self.ranges)
        

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

    @abstractmethod
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
        pass


class SimpleFunctionWrapper:
    """
    Class that can be used to wrap a TestFunction instance. Wrapped functions
    will be callable by providing each parameter as a separate argument,
    instead of in a single numpy array.

        func = Rosenbrock()
        simple_func = LeviNmbr13(func)
        y = simple_func(1, 2)

    Args:
        function: TestFunction to be wrapped

    Raises:
        Exception: SimpleFunctionWrapper can only wrap instances of the
            TestFunction class
    """

    def __init__(self, function):
        if not isinstance(function, TestFunction):
            raise Exception("SimpleFunctionWrapper can only wrap instances of"
                            "the TestFunction class.")
        self.function = function

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped testfunction through an altered interface. Instead
        of providing the data as a numpy array, the data is provided as a
        separate argument for each parameter. These parameters can be given as
        a numpy array, to evaluate multiple datapoints at the same time.

            func = Rosenbrock()
            simple_func = LeviNmbr13(func)
            y = simple_func(1, 2)

        Args:
            *args: Each of the parameters for the function, provided as unnamed
                arguments. Parameters may be provided as numbers (float/int) or
                as numpy arrays of consistent length (allowing for the
                evaluation of multiple datapoints at the same time).
            derivative: If this boolean is False (default), the testfunction
                will be queried for its value. If it is True, the derivative
                of the function is queried (if existing).
            epsilon: leeway parameter that is added to all minima and
                subtracted from all maxima in the .check_ranges method. Default
                is 0.

        Returns:
            If input was provided as numpy arrays or the output of the wrapped
            TestFunction is multi-dimensional, a numpy.ndarray of shape
            (nDatapoints, ?) containing the function evaluations will be
            returned. If data was provided as numbers, the result of the
            testfunction evaluation will be returned as a number or a list
            (depending on the dimensionality of the function output).

        Raises:
            Exception: Number of provided unnamed arguments should
                match the dimensionality of the wrapped TestFunction.
        """
        # Check dimensionality of the input
        if len(args) != len(self.function.ranges):
            raise Exception("Number of provided unnamed arguments should match"
                            "the dimensionality of the wrapped TestFunction.")
        # Construct input array for the wrapped TestFunction
        x = self._create_input_array(args)
        # Get valid keyword arguments
        kwargs = self._select_keyword_arguments(kwargs)
        # Evaluate function and change type/form before returning its result
        evaluation = self.function(x, **kwargs)
        if evaluation.shape == (1, 1):
            return evaluation[0, 0]
        return evaluation

    def _create_input_array(self, args):
        """
        Combine variable-separated input arguments into a single numpy array.

        Args:
            args: Tuple of numbers or numpy arrays, which should be combined
                into a single numpy array to be provided to a TestFunction's
                __call__ method.

        Returns:
            Numpy.ndarray of shape (nDatapoints, ?)
        """
        parameters = []
        for parameter in args:
            if isinstance(args, np.ndarray):
                parameter = parameter.flatten()
            parameters.append(parameter)
        x = np.hstack(parameters)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return x

    def _select_keyword_arguments(self, kwargs_dict):
        """
        Filter out elements of the provided dictionary with keys 'derivative'
        and 'epsilon'.

        Args:
            kwargs_dict: Dictionary of which the elements should be filtered.

        Returns:
            Dictionary containing only the entries of the input dictionary that
            have keys 'dictionary' and 'epsilon'. If some, or all, of them do
            not exist, no such key will appear in the output dictionary.
        """
        kwargs = {}
        for k in kwargs_dict:
            if k in ['derivative', 'epsilon']:
                kwargs[k] = kwargs_dict[k]
        return kwargs


class NoDerivativeError(Exception):
    """
    Exception indicating no derivative is known to queried testfunction
    """
    pass


class FunctionFeeder:
    """
    Function container to perform larger automated experiments

    The FunctionFeeder is a container class to which functions can be added. It
    operates as an iterator, so it can be used as the argument of a for loop,
    which will then feed the functions in the container one by one to the
    code in the loop.
    """

    def __init__(self):
        self.reset()

    def __len__(self):
        """
        Get the number of functions in the FunctionFeeder instance
        """
        return len(self.functions)

    def __iter__(self):
        """
        Get the iterable of functions to be used in e.g. for loops
        """
        return iter(self.functions)

    def reset(self):
        """
        Empty the feeder from all loaded functions
        """
        self.functions = []

    def load_function_group(self, group, parameters=None):
        """
        Load functions from a specific set

        This method adds test functions to the FunctionFeeder that are
        belonging to a specific group of test functions. Currently implemented
        are the following groups:

            optimisation / optimization: Rastrigin, Rosenbrock, Beale, Booth,
                BukinNmbr6, Mayas, LeviNmbr13, Himmelblau, ThreeHumpCamel,
                Sphere, Ackley, Easom, Linear
            posterior: Cosine, Block, Bessel, ModifiedBessel, Eggbox,
                MultivariateNormal, GaussianShells, Linear, Reciprocal,
                BreitWigner
            with_derivative: Rastrigin, Sphere, Cosine, Bessel, ModifiedBessel,
                Reciprocal, BreitWigner
            no_derivative: Rosenbrock, Beale, Booth, BukinNmbr6, Matyas,
                LeviNmbr13, Himmelblau, ThreeHumpCamel, Ackley, Easom, Block,
                Eggbox, MultivariateNormal, GaussianShells,
                Linear
            bounded: Rastrigin, Beale, Booth, BukinNmbr6, Matyas, LeviNmbr13,
                Himmelblau, ThreeHumpCamel, Ackley, Easom, Bessel,
                ModifiedBessel, Eggbox, MultivariateNormal, GaussianShells,
                Linear, Reciprocal, BreitWigner
            unbounded: Rosenbrock, Sphere, Block

        All functions are loaded with default configuration, unless
        configuration is set through the parameters argument.

        Args:
            groups: Name of the test function group to load. See above for all
                implemented groups and the functions for each of these groups.
                It is also possible to provide a list of names, which could
                load the intersection of the provided groups.
            parameters: Dictionary containing all configuration variables
                for all test functions that should differ from the default
                setting. Parameters are provided on a per-function basis via
                their classname. Any parameter and function not defined in this
                dictionary is configured with its default value(s). For
                instance: `{'Rastrigin': {'dimensionality':4}}`.

        Raises:
            Exception: Group '?' not known
        """
        # Define functions for group
        function_names = {
            'optimisation': [
                'Rastrigin', 'Rosenbrock', 'Beale', 'Booth', 'BukinNmbr6',
                'Matyas', 'LeviNmbr13', 'Himmelblau', 'ThreeHumpCamel',
                'Sphere', 'Ackley', 'Easom', 'Linear', 'Reciprocal'
            ],
            'posterior': [
                'Cosine', 'Block', 'Bessel', 'ModifiedBessel', 'Eggbox',
                'MultivariateNormal', 'GaussianShells', 'Linear', 'BreitWigner'
            ],
            'with_derivative': [
                'Rastrigin', 'Sphere', 'Cosine', 'Bessel', 'ModifiedBessel',
                'Reciprocal', 'BreitWigner'
            ],
            'no_derivative': [
                'Rosenbrock', 'Beale', 'Booth', 'BukinNmbr6', 'Matyas',
                'LeviNmbr13', 'Himmelblau', 'ThreeHumpCamel', 'Ackley',
                'Easom', 'Block', 'Eggbox', 'MultivariateNormal',
                'GaussianShells', 'Linear'
            ],
            'bounded': [
                'Rastrigin', 'Beale', 'Booth', 'BukinNmbr6', 'Matyas',
                'LeviNmbr13', 'Himmelblau', 'ThreeHumpCamel', 'Ackley',
                'Easom', 'Bessel', 'ModifiedBessel', 'Eggbox',
                'MultivariateNormal', 'GaussianShells', 'Linear', 'Reciprocal',
                'BreitWigner'
            ],
            'unbounded': ['Rosenbrock', 'Sphere', 'Block']
        }
        function_names['optimization'] = function_names['optimisation']
        # Check if provided function names are known
        if isinstance(group, str):
            if group not in function_names:
                raise Exception("Group '{}' not known".format(group))
        elif isinstance(group, list):
            for g in group:
                if g not in function_names:
                    raise Exception("Group '{}' not known".format(g))
        else:
            raise Exception("Group should be a string or a list of strings")
        # Create list of function names to load
        load = None
        if isinstance(group, str):
            load = function_names[group]
        else:
            for g in group:
                if load is not None:
                    load = [func for func in load if func in function_names[g]]
                else:
                    load = function_names[g]
        # Loop over function names and load each function
        if parameters is None:
            parameters = {}
        for name in load:
            if name not in parameters:
                self.load_function(name)
            else:
                self.load_function(name, parameters[name])

    def load_function(self, functionname, parameters=None):
        """
        Loads a function by function name and configures the parameters of this
        function.

        Args:
            functionname: Classname of the function that needs to be loaded
                and added to the FunctionFeeder container.
            parameters: Dictionary containing the parameters to configure
                and the values that these parameters should take. Any parameter
                not set in this dictionary will be set to its default value.

        Raises:
            Exception: Function name '?' unknown.
            Exception: Cannot load a function that is not derived from the
                TestFunction base class.
        """
        # Initialise testfunction
        if functionname not in globals():
            raise Exception("Function name '{}' unknown".format(functionname))
        f = globals()[functionname]()
        if not isinstance(f, TestFunction):
            raise Exception("""Cannot load a function that is not derived from
                               the TestFunction base class.""")
        # Configure testfunction
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise Exception("Parameters should be provided as a dictionary.")
        f = globals()[functionname](**parameters)
        # Store testfunction
        self.add_function(f)

    def add_function(self, function):
        """
        Add a configured function to the FunctionFeeder instance

        Args:
            function: Test function (instance of a class with the TestFunction
                class as its base class) that should be added to the
                FunctionFeeder.

        Raises:
            Exception: Cannot load a function that is not derived from the
                TestFunction base class.
        """
        if not isinstance(function, TestFunction):
            raise Exception("""Cannot load a function that is not derived from
                               the TestFunction base class.""")
        self.functions.append(function)
    
    def check_name_uniqueness(self, correct_duplicates=True):
        """
        Check if the names of the TestFunctions in the loader are all unique.
        If not, correct them if requested.

        Args:
            correct_duplicates: Boolean indicating whether or not duplicate
                names should be corrected. If `True` (default), they will be
                appended with '_config*', where '*' is an increasing number.
        """
        known_names = []
        corrections = {}
        # Get all duplicate names
        for func in self.functions:
            if func.name in known_names and func.name not in corrections.keys():
                corrections[func.name] = 1
            known_names.append(func.name)
        del(known_names)
        # Raise exception if correction not requested
        if len(corrections) > 0:
            if not correct_duplicates:
                raise Exception("TestFunction names not unique. Duplicates: {}.".format(', '.join(corrections.keys())))
            # Correct names
            for i, func in enumerate(self.functions):
                if func.name in corrections:
                    new_name = func.name+'_config'+str(corrections[func.name])
                    corrections[func.name] += 1
                    self.functions[i].name = new_name


class Rastrigin(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Rastrigin_function

    The application range of this function is -5.12 to 5.12 for each of the
    input dimensions.

    Args:
        dimensionality: Number of input dimensions the function should take.
    """

    def __init__(self, dimensionality=2, **kwargs):
        self.ranges = self.construct_ranges(dimensionality, -5.12, 5.12)
        self.a = 10
        super(Rastrigin, self).__init__(**kwargs)

    def _evaluate(self, x):
        n = len(self.ranges)
        y = self.a * n
        for i in range(n):
            y += np.power(x[:, i], 2) - self.a * np.cos(2 * np.pi * x[:, i])
        return y.reshape(-1, 1)

    def _derivative(self, x):
        return 2 * x + 2 * np.pi * self.a * np.sin(2 * np.pi * x)


class Rosenbrock(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Rosenbrock_function

    This function has a dynamic dimensionality and is its application range is
    unbounded. There is no derivative defined.
    """

    def __init__(self, dimensionality=2, **kwargs):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rosenbrock function has to
                            be >=2.""")
        self.ranges = self.construct_ranges(dimensionality, -np.inf, np.inf)
        super(Rosenbrock, self).__init__(**kwargs)

    def _evaluate(self, x):
        n = len(self.ranges)
        y = 0
        for i in range(n - 1):
            y += (100 * np.power(x[:, i + 1] - np.power(x[:, i], 2), 2) +
                  np.power(1 - x[:, i], 2))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Beale(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range of -4.5 to 4.5
    for both dimensions. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-4.5, 4.5], [-4.5, 4.5]]
        super(Beale, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = (np.power(1.5 - x[:, 0] + x[:, 0] * x[:, 1], 2) +
             np.power(2.25 - x[:, 0] + x[:, 0] * np.power(x[:, 1], 2), 2) +
             np.power(2.625 - x[:, 0] + x[:, 0] * np.power(x[:, 1], 3), 2))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Booth(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function bounded by -10 and 10 for both input
    dimensions. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-10, 10], [-10, 10]]
        super(Booth, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = np.power(x[:, 0] + 2 * x[:, 1] - 7, 2) + np.power(
            2 * x[:, 0] + x[:, 1] - 5, 2)
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class BukinNmbr6(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range bounded by -15
    and -5 for the first input variable and -3 and 3 for the second input
    variable. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-15, -5], [-3, 3]]
        super(BukinNmbr6, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = 100 * np.sqrt(
            np.abs(x[:, 1] - 0.01 * np.power(x[:, 0], 2)) +
            0.01 * np.abs(x[:, 0] + 10))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Matyas(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range bounded by -10
    and 10 for both input variables. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-10, 10], [-10, 10]]
        super(Matyas, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = 0.26 * (np.power(x[:, 0], 2) +
                    np.power(x[:, 1], 2)) - 0.48 * x[:, 0] * x[:, 1]
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class LeviNmbr13(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range boundedd by -10
    and 10 for both input variables. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-10, 10], [-10, 10]]
        super(LeviNmbr13, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = (np.power(np.sin(3 * np.pi * x[:, 0]), 2) +
             np.power(x[:, 0] - 1, 2) *
             (1 + np.power(np.sin(3 * np.pi * x[:, 1]), 2)) +
             np.power(x[:, 1] - 1, 2) *
             (1 + np.power(np.sin(2 * np.pi * x[:, 1]), 2)))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Himmelblau(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for both input variables. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-5, 5], [-5, 5]]
        super(Himmelblau, self).__init__(**kwargs)

    def _evaluate(self, x):
        return (np.power(np.power(x[:, 0], 2) + x[:, 1] - 11, 2) +
                np.power(x[:, 0] + np.power(x[:, 1], 2) - 7, 2)).reshape(
                    -1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class ThreeHumpCamel(TestFunction):
    """
    Testfunction as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for both input variables. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-5, 5], [-5, 5]]
        super(ThreeHumpCamel, self).__init__(**kwargs)

    def _evaluate(self, x):
        return (2.0 * np.power(x[:, 0], 2) - 1.05 * np.power(x[:, 0], 4) +
                np.power(x[:, 0], 6) / 6.0 + x[:, 0] * x[:, 1] +
                np.power(x[:, 1], 2)).reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Sphere(TestFunction):
    """
    Testfunction that returns the squared euclidean distance on evaluation.

    Testfunction following the formula:

        y = sum_{i=1}^{N} x_i^2

    The derivative of this function is implemented as

        y' = 2*x

    The number of input dimensions for this function is configurable at
    initialisation of and instance of this class. For each of these dimensions
    the application range is unbounded.
    """

    def __init__(self, dimensionality=3, **kwargs):
        self.ranges = self.construct_ranges(dimensionality, -np.inf, np.inf)
        super(Sphere, self).__init__(**kwargs)

    def _evaluate(self, x):
        return np.sum(np.power(x, 2), axis=1).reshape(-1, 1)

    def _derivative(self, x):
        return (2 * x).reshape(-1, 1)


class Ackley(TestFunction):
    """
    Ackley function as defined by
    https://en.wikipedia.org/wiki/Ackley_function.

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for each of these dimensions. No derivative has been defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[-5, 5], [-5, 5]]
        super(Ackley, self).__init__(**kwargs)

    def _evaluate(self, x):
        a = -20 * np.exp(
            -0.2 * np.sqrt(0.5 *
                           (np.power(x[:, 0], 2) + np.power(x[:, 1], 2))))
        f = np.cos(2 * np.pi * x[:, 0])
        g = np.cos(2 * np.pi * x[:, 1])
        b = -np.exp(0.5 * (f + g))
        y = a + b + np.exp(1) + 20
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Easom(TestFunction):
    """
    Easom function as defined by
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    This is a 2-dimensional function with an application range bounded by
    a box between -x and x for both dimensions, where x can be defined by
    the user (100 is default). No derivative has been defined.

    Args:
        absolute_range: Absolute value of the boundaries of the application
            range for the function. Both dimensions will be bounded to the
            range [-1 * absolute_range, absolute_range]. Is set to 100 by
            default, as is customary for this function.
    """

    def __init__(self, absolute_range=100, **kwargs):
        self.ranges = [[-absolute_range, absolute_range],
                       [-absolute_range, absolute_range]]
        super(Easom, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = (-1 * np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.exp(
            -1 *
            (np.power(x[:, 0] - np.pi, 2) + np.power(x[:, 1] - np.pi, 2))))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Cosine(TestFunction):
    """
    1-D cosine function meant for posterior sampling.

        f(x) = cos(x) + 1

    The ranges have been set to [-4*pi, 4*pi].
    """

    def __init__(self):
        self.ranges = [[-4 * np.pi, 4 * np.pi]]
        super(Cosine, self).__init__(**kwargs)

    def _evaluate(self, x):
        return (np.cos(x) + 1).reshape(-1, 1)

    def _derivative(self, x):
        return (-np.sin(x) + 1).reshape(-1, 1)


class Block(TestFunction):
    """
    Multi-dimensional block function.

    A function that is `global_value` everywhere except in the block spanned by
    [-`block_size`, `block_size`] in every dimension, where it takes on
    `block_value`.

    The application range of this function is set to be -10 to 10 for each of
    the input dimensions. No derivative has been defined.

    Args:
        dimensionality: Number of dimensions of the function. By default set
            to 3.
        block_size: Defines the ranges in which the block function should take
            on another value than the global default. Ranges are set for each
            dimension as [-block_size, block_size]. Default: 1.
        block_value: Value that the function takes *inside* the ranges spanned
            by block_size. Default: 1.
        global_value: Value that the function takes outside of the ranges
            spanned by block_size. Default: 0.

    """

    def __init__(self,
                 dimensionality=3,
                 block_size=1,
                 block_value=1,
                 global_value=0,
                 **kwargs):
        self.dimensionality = dimensionality
        self.block_size = block_size
        self.block_value = block_value
        self.global_value = global_value
        self.ranges = self.construct_ranges(dimensionality, -10, 10)
        super(Block, self).__init__(**kwargs)

    def _evaluate(self, x):
        boundary = np.array([self.block_size] * self.dimensionality)
        inidx = np.all((-1 * boundary <= x) & (x <= boundary), axis=1)
        y = self.global_value + (self.block_value - self.global_value) * inidx
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Bessel(TestFunction):
    """
    Bessel function of the first kind.

    Depending on the chosen function configuration, the computation of the
    function is performed using the jv function from scipy (if `fast` is set
    to False):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html
    or the j0 and j1 function (if `fast` is set to True):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j0.html,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j1.html.

    To make sampling from this function possible, 0.5 is added to the evaluated
    function value.

    The fast version of this function is an approximation of the true Bessel
    function.

    This is a 1-dimensional function with a range set to -100 to 100.

    Args:
        fast: Boolean indicating which set of Bessel function implementations
            to use. See above for more information.
    """

    def __init__(self, fast=False, **kwargs):
        self.ranges = [[-100, 100]]
        self.fast = bool(fast)
        super(Bessel, self).__init__(**kwargs)

    def _evaluate(self, x):
        if not self.fast:
            return special.jv(0, x) + 0.5
        return special.j0(x) + 0.5

    def _derivative(self, x):
        if not self.fast:
            return special.jv(1, x)
        return special.j1(x)


class ModifiedBessel(TestFunction):
    """
    Modified Bessel function of the first kind.

    Depending on the chosen function configuration, the computation of the
    function is performed using the jv function from scipy (if `fast` is set
    to False):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html
    or the j0 and j1 function (if `fast` is set to True):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0.html,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k1.html.

    The fast version of this function is an approximation of the true modified
    bessel function.

    This is a 1-dimensional function with a range set to 0 to 10.

    Args:
        fast: Boolean indicating which set of Bessel function implementations
            to use. See above for more information.
    """

    def __init__(self, fast=False, **kwargs):
        self.ranges = [[0, 10]]
        self.fast = bool(fast)
        super(ModifiedBessel, self).__init__(**kwargs)

    def _evaluate(self, x):
        if not self.fast:
            return special.kv(0, x)
        return special.k0(x)

    def _derivative(self, x):
        if not self.fast:
            return special.kv(1, x)
        return special.k1(x)


class Eggbox(TestFunction):
    """
    The Eggbox likelihood function as defined in the multinest paper
    https://arxiv.org/pdf/0809.3437.pdf:

        L(x, y) = exp(2 + cos(x/2)*cos(y/2))^5

    This is a 2-dimensional function bounded 0 and 10*pi in each dimension. No
    derivative is defined.
    """

    def __init__(self, **kwargs):
        self.ranges = [[0, 10 * np.pi], [0, 10 * np.pi]]
        super(Eggbox, self).__init__(**kwargs)

    def _evaluate(self, x):
        y = np.exp(
            np.power(2 + np.cos(x[:, 0] / 2.0) * np.cos(x[:, 1] / 2.0), 5))
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class MultivariateNormal(TestFunction):
    """
    Multivariate normal distribution

    The dimensionality of the function is determined by the determined based on
    the provided covariance matrix. The range of the function is [-10, 10]
    for each of the dimensions.

    Args:
        covariance: 2-dimension list or numpy array representing the covariance
            matrix to use. By default is is set to the 2-dimensional unit
            matrix, making the function 2-dimensional.
    """

    def __init__(self, covariance=None, **kwargs):
        if covariance is None:
            covariance = np.identity(2)
        self.covariance = covariance
        n_dim = len(covariance)
        self.ranges = self.construct_ranges(n_dim, -10, 10)
        super(MultivariateNormal, self).__init__(**kwargs)

    def _evaluate(self, x):
        mu = np.zeros(len(self.covariance))
        y = stats.multivariate_normal.pdf(x, mu, self.covariance)
        return y.reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class GaussianShells(TestFunction):
    """
    The Gaussian Shells likelihood function as defined in the multinest paper
    https://arxiv.org/pdf/0809.3437.pdf:

        L(x, y) = circ(x, c_1, r_1, w_1) + circ(x, c_2, r_2, w_2)
        circ(x, c, r, w) = exp(-(|x-c|-r)^2/(2*w^2)) / sqrt(2*pi*w^2)

    where x and c are vectors in a flat 2-dimensional space, making this
    testfunction 2-dimensional. The ranges of this function are set to
    [-10, 10] for both input dimensions.

    This is a 2-dimensional function bounded 0 and 10*pi in each dimension. No
    derivative is defined.

    Args:
        c_1: Numpy array or list with two entries, defining the center of the
            first gaussian shell. It is set to [2.5, 0] by default.
        r_1: Radius of the first gaussian shell. It is 2.0 by default.
        w_1: Standard deviation of the first gaussian shell. By default this
            value is 0.1.
        c_2: Numpy array or list with two entries, defining the center of the
            second gaussian shell. It is set to [2.5, 0] by default.
        r_2: Radius of the second gaussian shell. It is 2.0 by default.
        w_2: Standard deviation of the second gaussian shell. By default this
            value is 0.1.
    """

    def __init__(self,
                 c_1=[2.5, 0],
                 r_1=2.0,
                 w_1=0.1,
                 c_2=[-2.5, 0],
                 r_2=2.0,
                 w_2=0.1,
                 **kwargs):
        self.c_1 = np.array(c_1)
        self.r_1 = r_1
        self.w_1 = w_1
        self.c_2 = np.array(c_2)
        self.r_2 = r_2
        self.w_2 = w_2
        self.ranges = [[-10, 10], [-10, 10]]
        super(GaussianShells, self).__init__(**kwargs)

    def _shell(self, x, c, r, w):
        return (np.exp(-1 * np.power(np.linalg.norm(x - c, axis=1) - r, 2) /
                       (2 * w * w)) / np.sqrt(2 * np.pi * w * w))

    def _evaluate(self, x):
        shell_1 = self._shell(x, self.c_1, self.r_1, self.w_1)
        shell_2 = self._shell(x, self.c_2, self.r_2, self.w_2)
        return (shell_1 + shell_2).reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError


class Linear(TestFunction):
    """
    Test function defined by:

        sum_i |x_i|

    The application range of this function is -10 to 10 for each of the input
    dimensions. No derivative is defined.

    Args:
        dimensionality: Number of dimensions for input of the function. By
            default this argument is set to 2.
    """

    def __init__(self, dimensionality=2, **kwargs):
        self.ranges = self.construct_ranges(dimensionality, -10, 10)
        super(Linear, self).__init__(**kwargs)

    def _evaluate(self, x):
        return np.sum(np.abs(x), 1).reshape(-1, 1)

    def _derivative(self, x):
        raise NoDerivativeError()


class Reciprocal(TestFunction):
    """
    Test function defined by

        prod_i x_i^(-1)

    The application range of this function is 0.001 to 1 for each fo the input
    dimensions. No derivative is defined.

    Args:
        dimensionality: Number of dimensions for input of the function. By
            default this argument is set to 2.
    """

    def __init__(self, dimensionality=2, **kwargs):
        self.ranges = self.construct_ranges(dimensionality, 0.001, 1)
        super(Reciprocal, self).__init__(**kwargs)

    def _evaluate(self, x):
        return np.prod(np.power(x, -1), 1).reshape(-1, 1)

    def _derivative(self, x):
        dimensionality = self.get_dimensionality()
        derivative = -1 * np.ones((len(x), dimensionality)) * self._evaluate(x)
        for d in range(dimensionality):
            derivative[:, d] *= np.power(x[:, d], -1)
        return derivative


class BreitWigner(TestFunction):
    """
    Test function defined by
    https://en.wikipedia.org/wiki/Relativistic_Breit–Wigner_distribution.

    The application range of this function is 0 to 100. No derivative is
    defined.

    Args:
        m: Configuration parameter of the distribution. In terms of physics
            it corresponds to the mass of the particle creating the
            resonance that is shaped like the Breit-Wigner distribution. Set
            to 50 by default.
        width: Configuration parameter of the distribution. In terms of physics
            it corresponds to the decay width of the particle of the resonance.
            Set to 15 by default.
    """

    def __init__(self, m=50, width=15, **kwargs):
        self.m = m
        self.width = width
        self.ranges = [[0, 100]]
        super(BreitWigner, self).__init__(**kwargs)

    def _k(self):
        return (2*np.sqrt(2)*self.m*self.width*self._gamma()
                / (np.pi * np.sqrt(self.m**2 + self._gamma())))

    def _gamma(self):
        return np.sqrt(self.m**2 * (self.m**2 + self.width**2))

    def _evaluate(self, x):
        return self._k() / (np.power(np.power(x, 2) - self.m**2, 2)
                            + self.m**2 * self.width**2)

    def _derivative(self, x):
        return (-4*self._k()*x*(np.power(x, 2) - self.m**2)
                / np.power(self.width**2 * self.m**2
                           + np.power(np.power(x, 2) - self.m**2, 2), 2))
