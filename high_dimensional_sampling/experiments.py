import os
import getpass
import pandas as pd
import yaml

from .utils import get_time, get_datetime, create_unique_folder, benchmark_matrix_inverse, benchmark_sha_hashing
from .methods import Sampler
from .functions import FunctionCallCounter


class SamplingExperiment:
    def __init__(self, method=None, location=None, overwrite=False):
        if not isinstance(method, Sampler):
            raise Exception("SamplingExperiments should be provided an instance of a class derived from the methods.Sampler class.")
        self.method = method
        self.logger = None
    
    def run(self, function, path, log_data=True, finish_line = None):
        # Test if function is a TestFunction instance
        # TODO
        # Create logger, which automatically creates the logging location
        self.logger = SamplingLogger(path, (type(function).__name__).lower())
        # Log experiment
        self.logger.log_experiment(self, function)
        # Store testfunction with a FunctionCallCounter wrapped around it for
        # logging purposes
        self.function = FunctionCallCounter(function)
        # Initialise method and get first queries
        self.method.function = self.function
        # Perform sampling as long as procedure is not finished
        is_finished = False
        self.N_sampled = 0
        while not is_finished:
            self.logger.method_calls += 1
            # As long as the experiment is not finished, sample data
            t_start = get_time()
            data = self.method.sample()
            dt = get_time() - t_start
            # Log method call
            N = len(data)
            self.N_sampled += N
            self.logger.log_method_calls(dt, self.N_sampled, N)
            # Log data
            if log_data:
                self.logger.log_samples(data)
            # Log function calls and reset counter
            self.logger.log_function_calls(self.function)
            self.function.reset()
            # Update is_finished conditional
            is_finished = self.method.is_finished()
            if isinstance(finish_line, int):
                is_finished = is_finished or (self.N_sampled >= finish_line)
        # Clean up by closing all handles
        self.logger.close_handles()


class SamplingLogger:
    def __init__(self, path, prefered_subfolder, initialise_handles=True):
        self.path = create_unique_folder(path, prefered_subfolder)
        self.method_calls = 0
        if initialise_handles:
            self.create_handles()
    
    def __del__(self):
        self.close_handles()

    def create_handles(self):
        self.handle_samples = open(self.path + os.sep + "samples.csv", "w")
        self.handle_method_calls = open(self.path + os.sep + "methodcalls.csv", "w")
        self.handle_function_calls = open(self.path + os.sep + "functioncalls.csv", "w")
        self.handle_experiment = open(self.path + os.sep + "experiment.csv", "w")
    
    def close_handles(self):
        # Close all handles of the files, after looking if they exist
        handles = ["samples", "method_calls", "function_calls", "experiment"]
        for handle in handles:
            if hasattr(self, handle):
                getattr(self, handle).close()        

    def reset(self):
        self.close_handles()
        self.method_calls = 0

    def log_samples(self, x):
        n_datapoints = len(x)
        points = x.astype(str).tolist()
        for i in range(n_datapoints):
            line = [self.method_calls].extend(points[i])
            self.handle_samples.write(','.join(line) + "\n")
    
    def log_method_calls(self, dt, size_total, size_generated):
        line = [dt, size_total, size_generated]
        self.handle_method_calls.write(','.join(line) + "\n")

    def log_function_calls(self, function):
        for time, derivative in zip(function.counter_time, function.counter_derivatives):
            line = [self.method_calls, time, derivative]
            self.handle_function_calls(','.join(line) + "\n")

    def log_hardware(self):
        raise NotImplementedError

    def log_experiment(self, experiment, function):
        info = {}
        # Get meta data of experiment
        info['meta'] = {
            'timestamp': get_datetime(),
            'user': getpass.getuser(),
            'benchmark': {
                'matrix_inversion': benchmark_matrix_inverse(),
                'sha_hashing': benchmark_sha_hashing()
            } 
        }
        # Get properties of function
        info['function'] = {
            'name': type(function).__name__,
            'properties': vars(function)
        }
        # Get properties of experiment
        info['experiment'] = {
            'name': type(experiment).__name__,
            'properties': vars(experiment)
        }
        # Convert information to yaml and write to file
        yaml.dump(info, self.handle_experiment, default_flow_style=False)
