import os
import getpass
import pandas as pd
import yaml
import copy

from .utils import get_time, get_datetime, create_unique_folder, benchmark_matrix_inverse, benchmark_sha_hashing
from .methods import Method
from .functions import TestFunction


class Experiment:
    """
    Class to perform experiments on Methods with TestFunctions

    This class allows to test sampling methods implemented as a derived class
    from the methods.Method class by letting it work on a TestFunction derived
    class instance. It automatically takes care of logging (through a Logger
    instance) and sanity checks.

    Args:
        method: An instance of a Method derived class that needs to be tested
            in this experiment.
        path: Path to which the experiment should write its logs.
    """

    def __init__(self, method, path):
        if not isinstance(method, Method):
            raise Exception("SamplingExperiments should be provided an instance of a class derived from the methods.Sampler class.")
        self.path = path
        self.method = method
        self.logger = None
    
    def run(self, function, log_data=True, finish_line=10000):
        """
        Run the experiment.

        Calling this method will run the experiment on the provided function.
        It will continue to run as long as the method being tested in this
        experiment is not finished (checked through its is_finished method)
        or a specified number of sampled datapoints is reached (configured
        via the finish_line argument of this method).

        Args:
            function: Function to run the experiment with. This should be an
                instance of a class with the functions.TestFunction class as
                base class.
            log_data: Boolean indicating if the sampled data should be logged
                as well. It is set to True by default.
            finish_line: If the total sampled data set reaches or exceeds this
                size, the experiment is stopped. This is a hard stop, not a
                stopping condition that has to be met: if the method being
                tested indicates it is finished, the experiment will be
                stopped, regardless of the size of the sampled data set. The
                finish_line is set to 10,000 by default. If set to None, the
                experiment will continue to run until the method indicates 
                it is finished.
        
        Raises:
            Exception: Provided function should have functions.TestFunction as
                base class.
        """
        # Test if function is a TestFunction instance
        if not isinstance(function, TestFunction):
            raise Exception("Provided function should have functions.TestFunction as base class.")
        # Create logger, which automatically creates the logging location
        self.logger = Logger(self.path, (type(function).__name__).lower())
        # Log experiment
        self.logger.log_experiment(self, function)
        # Store testfunction
        self.function = function
        # Initialise method and get first queries
        self.method.function = self.function
        # Perform sampling as long as procedure is not finished
        is_finished = False
        self.N_sampled = 0
        while not is_finished:
            self.logger.method_calls += 1
            # As long as the experiment is not finished, sample data
            t_start = get_time()
            X, y = self.method(self.function)
            dt = get_time() - t_start
            # Log method call
            N = len(X)
            self.N_sampled += N
            self.logger.log_method_calls(dt, self.N_sampled, N)
            # Log data
            if log_data:
                self.logger.log_samples(X, y)
            # Log function calls and reset counter
            self.logger.log_function_calls(self.function)
            self.function.reset()
            # Update is_finished conditional
            is_finished = self.method.is_finished()
            if isinstance(finish_line, int):
                is_finished = is_finished or (self.N_sampled >= finish_line)
        # Delete the logger to close all handles
        del(self.logger)


class Logger:
    """
    Class that takes care of all logging of experiments.

    An instance of this class is automatically made and handled within the
    Experiment class.

    Args:
        path: Path to which logging results should be written. Within this
            folder each test function will get its own subfolder.
        prefered_subfolder: Name of the folder to be created in the logging
            path. The folder is created with the utils.create_unique_folder
            function, so naming conflicts will be automatically resolved.
    """

    def __init__(self, path, prefered_subfolder):
        self.path = create_unique_folder(path, prefered_subfolder)
        self.method_calls = 0
        self.create_samples_header = True
        self._create_handles()
    
    def __del__(self):
        """
        Closes all the opened handles at deletion of the instance.
        """
        handles = ["samples", "functioncalls", "methodcalls"]
        for handle in handles:
            if hasattr(self, 'handle_'+handle):
                getattr(self, 'handle_'+handle).close()

    def _create_handles(self):
        """
        Creates the file handles needed for logging. Created csv files also get
        their headers added if already possible.
        """
        self.handle_samples = open(self.path + os.sep + "samples.csv", "w")
        self.handle_functioncalls = open(self.path + os.sep + "functioncalls.csv", "w")
        self.handle_functioncalls.write('method_call_id,n_queried,dt,asked_for_derivative\n')
        self.handle_methodcalls = open(self.path + os.sep + "methodcalls.csv", "w")
        self.handle_methodcalls.write('method_call_id,dt,total_dataset_size,new_data_generated\n')

    def log_samples(self, x, y):
        """
        Log samples and their obtained function values from the test function.

        The data and their target values are written to the samples.csv file
        created at initialisation of the Logger object. As this is the first
        moment we know how many parameters the problem has, this function will
        create a header in this file as well if it is called for the first
        time.

        Args:
            x: numpy.ndarray of shape (nDatapoints, nVariables) containing the
                data to be logged.
            y: numpy.ndarray of shape (nDatapoints, nTargetVariables)
                containing the sampled function values of the test function.
        """
        # Create header
        if self.create_samples_header:
            header = ['method_call_id']
            header += ['x'+str(i) for i in range(len(x[0]))]
            header += ['y'+str(i) for i in range(len(y[0]))]
            self.handle_samples.write(",".join(header)+"\n")
            self.create_samples_header = False
        # Create and write line
        n_datapoints = len(x)
        points = x.astype(str).tolist()
        labels = y.astype(str).tolist()
        for i in range(n_datapoints):
            line = [str(self.method_calls)]
            line += points[i]
            line += labels[i]
            self.handle_samples.write(','.join(line) + "\n")
    
    def log_method_calls(self, dt, size_total, size_generated):
        """
        Log a method call to the methodscalls.csv file.

        Args:
            dt: Time in ms spend on the method call.
            size_total: Number of data points sampled in total for all
                method calls so far. This should include the data points
                sampled in the iteration that is currently sampled.
            size_generated: Number of data points sampled in this specific
                method call.
        """
        line = [int(self.method_calls), dt, int(size_total), int(size_generated)]
        line = list(map(str, line))
        self.handle_methodcalls.write(','.join(line) + "\n")

    def log_function_calls(self, function):
        """
        Log the number of calls to the test function and whether or not it is
        queried for a derivative.

        Function calls will be logged in the functioncalls.csv file.

        Args:
            function: Test function that was used in an experiment iteration.
                This test function should be a class with
                functions.TestFunction as its base class.
        """
        for entry in function.counter:
            line = [int(self.method_calls), int(entry[0]), float(entry[1]), bool(entry[2])]
            line = list(map(str, line))
            self.handle_functioncalls.write(','.join(line) + "\n")

    def log_experiment(self, experiment, function):
        """
        Log the setup and the function set up to a .yaml-file in order to
        optimize reproducability.

        This method should be called *before* the first experiment iteration.

        Args:
            experiment: Experiment to be run, containing the method to be
                tested (which needs to be provided at initialisation).
            function: Test function that was used in an experiment iteration.
                This test function should be a class with
                functions.TestFunction as its base class.
        """
        with open(self.path + os.sep + "experiment.yaml", "w") as handle:
            info = {}
            # Get meta data of experiment
            info['meta'] = {
                'datetime': str(get_datetime()),
                'timestamp': str(get_time()),
                'user': getpass.getuser(),
                'benchmark': {
                    'matrix_inversion': benchmark_matrix_inverse(),
                    'sha_hashing': benchmark_sha_hashing()
                } 
            }
            # Get properties of function
            info['function'] = {
                'name': type(function).__name__,
                'properties': copy.copy(vars(function))
            }
            del(info['function']['properties']['counter'])
            # Get properties of experiment
            info['method'] = {
                'name': type(experiment.method).__name__,
                'properties': {}
            }
            for prop in experiment.method.store_parameters:
                info['method']['properties'][prop] = getattr(experiment.method, prop)
            # Convert information to yaml and write to file
            yaml.dump(info, handle, default_flow_style=False)
