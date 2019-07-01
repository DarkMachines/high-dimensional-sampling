from abc import ABC, abstractmethod


class Method(ABC):
    """
    Abstract base class for all sampling methods

    All sampling methods subjected to an experiment should be derived from this
    class. It requires the implementation of the __init__, __call__ and
    is_finished methods.

    As it is an abstract base class, direct (i.e. not derived) instances of
    this class cannot exist.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialisation method for instances of the Method class.

        In this method anything can be put, but it should always at least
        define a property store_parameters, containing the configuration
        parameters of the method. The parameters indicated in this list are
        then automatically logged at the start of an experiment. If no such
        parameters exist, store_parameters should be an empty list.

        This method can be overwritten and contain input arguments, but for
        future compatibility all these input arguments should have defaults
        set.
        """
        self.store_parameters = []

    @abstractmethod
    def __call__(self, function):
        """
        Call for the sampling of more data points.

        This method queries the method to sample new data points (or a single
        new data point, whatever is more natural to the method). The function
        to be sampled is provided as an argument.

        Args:
            function: An instance of a test function derived from the
                TestFunction class. This function can be queried for values
                and derivatives by calling the function with function(data) and
                function(data, True) respectively.

        Returns:
            x: Sampled data in the form of a numpy.ndarray of shape
                (nDatapoints, nVariables).
            y: Function values for the samples datapoints of shape
                (nDatapoints, ?)
        """
        raise NotImplementedError

    @abstractmethod
    def is_finished(self):
        """
        Checks if the method is finished with sampling.

        This method is called at each iteration in the Experiment. When it
        returns True, the experiment is stopped. As such, it can be used as a
        check for convergence.

        Returns:
            A boolean that is True if the method is finished sampling. If this
            happens, the experiment in which this Method is tested will stop.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets all internal settings to the defaults
        """
        raise NotImplementedError
