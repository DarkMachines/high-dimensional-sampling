import os
import getpass
import pandas as pd
import yaml
import copy

from .utils import get_time, get_datetime, create_unique_folder, benchmark_matrix_inverse, benchmark_sha_hashing
from .methods import Sampler
from .functions import TestFunction


class SamplingExperiment:
    def __init__(self, method=None, path=None):
        if not isinstance(method, Sampler):
            raise Exception("SamplingExperiments should be provided an instance of a class derived from the methods.Sampler class.")
        self.path = path
        self.method = method
        self.logger = None
    
    def run(self, function, log_data=True, finish_line=None):
        # Test if function is a TestFunction instance
        if not isinstance(function, TestFunction):
            raise Exception("Provided function should be an instance of a class derived from functions.TestFunction.")
        # Create logger, which automatically creates the logging location
        self.logger = SamplingLogger(self.path, (type(function).__name__).lower())
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
        self.handle_functioncalls = open(self.path + os.sep + "functioncalls.csv", "w")
        self.handle_methodcalls = open(self.path + os.sep + "methodcalls.csv", "w")
        
    def close_handles(self):
        # Close all handles of the files, after looking if they exist
        handles = ["samples", "functioncalls", "methodcalls"]
        for handle in handles:
            if hasattr(self, 'handle_'+handle):
                getattr(self, 'handle_'+handle).close()        

    def reset(self):
        self.close_handles()
        self.method_calls = 0

    def log_samples(self, X, y):
        n_datapoints = len(X)
        points = X.astype(str).tolist()
        labels = y.astype(str).tolist()
        for i in range(n_datapoints):
            line = [str(self.method_calls)]
            line += points[i]
            line += [labels[i]]
            self.handle_samples.write(','.join(line) + "\n")
    
    def log_method_calls(self, dt, size_total, size_generated):
        line = [dt, int(size_total), int(size_generated)]
        line = list(map(str, line))
        self.handle_methodcalls.write(','.join(line) + "\n")

    def log_function_calls(self, function):
        for time, derivative in zip(function.counter_time, function.counter_derivatives):
            line = [int(self.method_calls), float(time), bool(derivative)]
            line = list(map(str, line))
            self.handle_functioncalls.write(','.join(line) + "\n")

    def log_experiment(self, experiment, function):
        with open(self.path + os.sep + "experiment.yaml", "w") as handle:
            info = {}
            # Get meta data of experiment
            info['meta'] = {
                'datetime': str(get_datetime()),
                'timestamp': str(get_time()),
                'user': getpass.getuser(),
                'benchmark': {
                    'matrix_inversion': 0,#benchmark_matrix_inverse(),
                    'sha_hashing': 0#benchmark_sha_hashing()
                } 
            }
            # Get properties of function
            info['function'] = {
                'name': type(function).__name__,
                'properties': copy.copy(vars(function))
            }
            del(info['function']['properties']['counter_time'])
            del(info['function']['properties']['counter_derivatives'])
            # Get properties of experiment
            info['method'] = {
                'name': type(experiment.method).__name__,
                'properties': vars(experiment.method)
            }
            # Convert information to yaml and write to file
            yaml.dump(info, handle, default_flow_style=False)
