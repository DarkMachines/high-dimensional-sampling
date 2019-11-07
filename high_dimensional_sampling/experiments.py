from abc import ABC, abstractmethod
import os
import getpass
import yaml
import copy
import numpy as np

from .utils import (get_time, get_datetime, create_unique_folder,
                    benchmark_matrix_inverse, benchmark_sha_hashing)
from .procedures import Procedure
from .functions import TestFunction


class Experiment(ABC):
    """
    Base class for performing experiments on Procedures with TestFunctions

    This class allows to test sampling procedures implemented as a derived
    class from the procedures.Procedure class by letting it work on a
    TestFunction derived class instance. It automatically takes care of logging
    (through a Logger instance) and sanity checks.

    Args:
        procedure: An instance of a Procedure derived class that needs to be
            tested in this experiment.
        path: Path to which the experiment should write its logs.
    """

    def __init__(self, procedure, path):
        if not isinstance(procedure, Procedure):
            raise Exception("""Experiments should be provided an instance of a
                class derived from the procedures.Procedure class.""")
        self.path = path
        self.procedure = procedure
        self.logger = None

    def _perform_experiment(self, function, log_data=True):
        """
        Run the experiment.

        Calling this method will run the experiment on the provided function.
        It will continue to run as long as the procedure being tested in this
        experiment is not finished (checked through its is_finished method)
        or a specified number of sampled datapoints is reached (configured
        via the finish_line argument of this procedure).

        Args:
            function: Function to run the experiment with. This should be an
                instance of a class with the functions.TestFunction class as
                base class.
            log_data: Boolean indicating if the sampled data should be logged
                as well. It is set to True by default.
            finish_line: If the total sampled data set reaches or exceeds this
                size, the experiment is stopped. This is a hard stop, not a
                stopping condition that has to be met: if the procedure being
                tested indicates it is finished, the experiment will be
                stopped, regardless of the size of the sampled data set. The
                finish_line is set to 10,000 by default. If set to None, the
                experiment will continue to run until the procedure indicates
                it is finished.

        Raises:
            Exception: Provided function should have functions.TestFunction as
                base class.
        """
        # Test if function is a TestFunction instance
        if not isinstance(function, TestFunction):
            raise Exception("""Provided function should have
                            functions.TestFunction as base class.""")
        # Test if the procedure can run on the provided test function
        if not self.procedure.check_testfunction(function):
            raise Exception("""Test function '{}' can not be used for '{}'. Ignoring and
                     continuing.""".format(function.name,
                                           type(self.procedure).__name__))
        # Start experiment
        print("Run experiment for '{}' on function '{}'...".format(
            type(self.procedure).__name__, function.name))
        self._event_start_experiment()
        # Setup logger
        self.logger = Logger(self.path, (function.name).lower())
        self.logger.log_experiment(self, function)
        self.logger.log_benchmarks()
        # Make function available both to the Experiment and the Procedure
        self.function = function
        self.procedure.reset()
        self.procedure.function = self.function
        # Perform sampling as long as procedure is not finished
        is_finished = False
        n_sampled = 0
        n_functioncalls = 0
        n_derivativecalls = 0
        t_experiment_start = get_time()
        while not is_finished:
            self.logger.procedure_calls += 1
            # Perform an procedure iteration and keep track of time elapsed
            t_start = get_time()
            x, y = self.procedure(self.function)
            dt = get_time() - t_start
            self._event_new_samples(x, y)
            # Log procedure call
            n = len(x)
            n_sampled += n
            self.logger.log_procedure_calls(dt, n_sampled, n)
            # Log sampled data
            if log_data:
                self.logger.log_samples(x, y)
            # Log function calls and reset the counter
            n_functioncalls += self.function.count_calls("normal")[1]
            n_derivativecalls += self.function.count_calls("derivative")[1]
            self.logger.log_function_calls(self.function)
            self.function.reset()
            # Check if the experiment has to stop and update the while
            # condition to control this.
            is_finished = (self.procedure.is_finished()
                           or self._stop_experiment(x, y))
        self._event_end_experiment()
        # Log result metrics
        t_experiment_end = get_time()
        metrics = {
            'time': (t_experiment_end - t_experiment_start),
            'n_functioncalls': n_functioncalls,
            'n_derivativecalls': n_derivativecalls
        }
        metrics = {**metrics, **self.make_metrics()}
        self.logger.log_results(metrics)
        # Delete the logger to close all handles
        del (self.logger)

    def _stop_experiment(self, x, y):
        """
        Uses the stopping criterion defined in the .run() method to determine
        if the experiment should be stopped.

        Args:
            x: Sampled data in the form of a numpy.ndarray of shape
                (nDatapoints, nVariables).
            y: Function values for the samples datapoints of shape
                (nDatapoints, ?)

        Returns:
            Boolean indicating if the experiment should be stopped (i.e. the
            stopping criterion is reached).
        """
        self.n_sampled += len(x)
        if self.n_sampled >= self.finish_line:
            return True
        return False

    @abstractmethod
    def make_metrics(self):
        """
        Creates metrics to report in experiment.yaml

        This is an abstract method and should be implemented in
        Experiment-specific classes derived from this one.

        Returns:
            Dictionary containing the metrics by name.
        """
        return {}

    @abstractmethod
    def _event_start_experiment(self):
        """
        Event that is run when a new experiment is started.

        This is an abstract method and should be implemented in
        Experiment-specific classes derived from this one.
        """
        pass

    @abstractmethod
    def _event_end_experiment(self):
        """
        Event that is run when an experiment is ended, but before the metrics
        are stored to the experiment.yaml file.

        This is an abstract method and should be implemented in
        Experiment-specific classes derived from this one.
        """
        pass

    @abstractmethod
    def _event_new_samples(self, x, y):
        """
        Event that is run when new samples are obtained from the specified
        procedure.

        This is an abstract method and should be implemented in
        Experiment-specific classes derived from this one.

        Args:
            x: Sampled data in the form of a numpy.ndarray of shape
                (nDatapoints, nVariables).
            y: Function values for the samples datapoints of shape
                (nDatapoints, ?)
        """
        pass

    def run(self, function, finish_line=1000, log_data=True):
        """
        Run the experiment on the provided test function.

        The experiment is stopped if the total number of sampled points reaches
        or exceeds the number defined in the `finish_line` argument.

        Args:
            function: Function to run the experiment with. This should be an
                instance of a class with the functions.TestFunction class as
                base class.
            finish_line: If the total sampled data set reaches or exceeds this
                size, the experiment is stopped. This is a hard stop, not a
                stopping condition that has to be met: if the procedure being
                tested indicates it is finished, the experiment will be
                stopped, regardless of the size of the sampled data set. The
                finish_line is set to 10,000 by default. If set to None, the
                experiment will continue to run until the procedure indicates
                it is finished.
            log_data: Boolean indicating if the sampled data should be logged
                as well. It is set to True by default.
        """
        self.finish_line = finish_line
        self.n_sampled = 0
        self._perform_experiment(function, log_data)


class OptimisationExperiment(Experiment):
    """
    Experiment class for optimisation experiments

    Implements automatic logging of best obtained result to the experiment.yaml
    file.
    """

    def _event_start_experiment(self):
        """
        Event that is run when a new experiment is started.
        """
        self.best_point = None

    def _event_end_experiment(self):
        """
        Event that is run when a experiment ends.
        """
        pass

    def _event_new_samples(self, x, y):
        """
        Event that is run when new samples are obtained from the specified
        procedure.

        This implementation checks all sampled points and their function values
        and stores the (x,y) pair that has the lowest function value.

        Args:
            x: Sampled data in the form of a numpy.ndarray of shape
                (nDatapoints, nVariables).
            y: Function values for the samples datapoints of shape
                (nDatapoints, ?)
        """
        for i in range(len(x)):
            if self.best_point is None:
                self.best_point = (x[i], y[i])
            elif y[i] < self.best_point[1]:
                self.best_point = (x[i], y[i])

    def make_metrics(self):
        """
        Creates metrics to report in results.yaml. Specifically: it reports the
        best found point (i.e. the point with the lowest function value).

        Returns:
            Dictionary containing the metrics by name.
        """
        if self.best_point is None:
            return {}
        return {
            'best_point': self.best_point[0].tolist(),
            'best_value': self.best_point[1].tolist()
        }


class OptimizationExperiment(OptimisationExperiment):
    """
    Experiment class for optimisation experiments. This class is a copy of
    OptimisationExperiment and its purpose is solely to support multiple
    language conventions.
    """
    pass


class PosteriorSamplingExperiment(Experiment):
    """
    Experiment class for posterior sampling experiments.
    """

    def _event_start_experiment(self):
        """
        Event that is run when a new experiment is started.
        """
        pass

    def _event_end_experiment(self):
        """
        Event that is run when a experiment ends.
        """
        pass

    def _event_new_samples(self, x, y):
        """
        Event that is run when new samples are obtained from the specified
        procedure.

        Args:
            x: Sampled data in the form of a numpy.ndarray of shape
                (nDatapoints, nVariables).
            y: Function values for the samples datapoints of shape
                (nDatapoints, ?)
        """
        pass

    def make_metrics(self):
        """
        Creates metrics to report in results.yaml. Specifically: it reports the
        best found point (i.e. the point with the lowest function value).

        Returns:
            Dictionary containing the metrics by name.
        """
        return {}


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
        self.basepath = path
        self.path = create_unique_folder(path, prefered_subfolder)
        self.procedure_calls = 0
        self.create_samples_header = True
        self._create_handles()

    def __del__(self):
        """
        Closes all the opened handles at deletion of the instance.
        """
        handles = ["samples", "functioncalls", "procedurecalls"]
        for handle in handles:
            if hasattr(self, 'handle_' + handle):
                getattr(self, 'handle_' + handle).close()

    def _create_handles(self):
        """
        Creates the file handles needed for logging. Created csv files also get
        their headers added if already possible.
        """
        self.handle_samples = open(self.path + os.sep + "samples.csv", "w")
        self.handle_functioncalls = open(
            self.path + os.sep + "functioncalls.csv", "w")
        self.handle_functioncalls.write(
            'procedure_call_id,n_queried,dt,asked_for_derivative\n')
        self.handle_procedurecalls = open(
            self.path + os.sep + "procedurecalls.csv", "w")
        self.handle_procedurecalls.write(
            'procedure_call_id,dt,total_dataset_size,new_data_generated\n')

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
            header = ['procedure_call_id']
            header += ['x' + str(i) for i in range(len(x[0]))]
            header += ['y' + str(i) for i in range(len(y[0]))]
            self.handle_samples.write(",".join(header) + "\n")
            self.create_samples_header = False
        # Create and write line
        n_datapoints = len(x)
        points = x.astype(str).tolist()
        labels = y.astype(str).tolist()
        for i in range(n_datapoints):
            line = [str(self.procedure_calls)]
            line += points[i]
            line += labels[i]
            self.handle_samples.write(','.join(line) + "\n")
        self.handle_samples.flush()

    def log_procedure_calls(self, dt, size_total, size_generated):
        """
        Log a procedure call to the procedurecalls.csv file.

        Args:
            dt: Time in ms spend on the procedure call.
            size_total: Number of data points sampled in total for all
                procedure calls so far. This should include the data points
                sampled in the iteration that is currently sampled.
            size_generated: Number of data points sampled in this specific
                procedure call.
        """
        line = [
            int(self.procedure_calls), dt,
            int(size_total),
            int(size_generated)
        ]
        line = list(map(str, line))
        self.handle_procedurecalls.write(','.join(line) + "\n")
        self.handle_procedurecalls.flush()

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
            line = [
                int(self.procedure_calls),
                int(entry[0]),
                float(entry[1]),
                bool(entry[2])
            ]
            line = list(map(str, line))
            self.handle_functioncalls.write(','.join(line) + "\n")
            self.handle_functioncalls.flush()

    def log_benchmarks(self):
        """
        Benchmark the machine with some simple benchmark algorithms (as
        implemented in the utils module).

        Results are stored in the base log path in the benchmarks.yaml file. If
        this file already exists, no benchmarks are run.
        """
        if os.path.exists(self.basepath + os.sep + "benchmarks.yaml"):
            return
        with open(self.basepath + os.sep + "benchmarks.yaml", "w") as handle:
            info = {}
            # Get meta data of experiment
            info['benchmarks'] = {
                'matrix_inversion': benchmark_matrix_inverse(),
                'sha_hashing': benchmark_sha_hashing(),
            }
            yaml.dump(info, handle, default_flow_style=False)

    def log_experiment(self, experiment, function):
        """
        Log the setup and the function set up to a .yaml-file in order to
        optimize reproducability.

        This method should be called *before* the first experiment iteration.

        Args:
            experiment: Experiment to be run, containing the procedure to be
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
            }
            # Get properties of function
            func_props = copy.copy(vars(function))
            for prop in func_props:
                if prop == 'name':
                    continue
                if isinstance(func_props[prop], np.ndarray):
                    func_props[prop] = func_props[prop].tolist()
            info['function'] = {
                'name': function.name,
                'testfunction': type(function).__name__,
                'properties': func_props
            }
            del (info['function']['properties']['counter'])
            # Get properties of experiment
            info['procedure'] = {
                'name': type(experiment.procedure).__name__,
                'properties': {}
            }
            info['experiment'] = {
                'type': experiment.__class__.__name__,
            }
            for prop in experiment.procedure.store_parameters:
                info['procedure']['properties'][prop] = getattr(
                    experiment.procedure, prop)
            # Convert information to yaml and write to file
            yaml.dump(info, handle, default_flow_style=False)

    def log_results(self, metrics):
        """
        Log the results of the experiment in the experiment.yaml file

        This method should be called *before* the first experiment iteration.

        Args:
            metrics: Dictionary containing the result metrics to store. Keys
                represent the name with which the values should be stored.
        """
        # Parse experiment yaml file and add results
        with open(self.path + os.sep + "experiment.yaml", 'r') as stream:
            experiment = yaml.safe_load(stream)
        experiment['results'] = metrics
        # Write new content to file
        with open(self.path + os.sep + "experiment.yaml", 'w') as handle:
            yaml.dump(experiment, handle, default_flow_style=False)
