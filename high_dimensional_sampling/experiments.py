import pandas as pd

from .utils import create_unique_folder
from .methods import Sampler
from .functions import FunctionCallCounter

class SamplingExperiment:
    def __init__(self, method=None, location=None, overwrite=False):
        if not isinstance(method, Sampler):
            raise Exception("SamplingExperiments should be provided an instance of a class derived from the methods.Sampler class.")
        self.method = method
        self.logger = SamplingLogger(location, overwrite)
    
    def __call__(self, function, path):
        # Store testfunction with a FunctionCallCounter wrapped around it for
        # logging purposes
        self.function = FunctionCallCounter(function)
        # Create dataframe for generated data
        columns = ["method_call", "dt", "derivative"]
        for i in range(len(self.function.function.ranges)):
            columns.append(str(i))
        self.data = pd.DataFrame(columns=columns)
        # Initialise method and get first queries
        self.method.initialise()
        # Perform sampling as long as procedure is not finished
        while not self.method.is_finished():
            raise NotImplementedError

    def evaluate(self, x, derivative):
        raise NotImplementedError

    def analyse(self):
        raise NotImplementedError


class SamplingLogger:
    def __init__(self, path, prefered_subfolder):
        self.path = create_unique_folder(path, prefered_subfolder)
        self.method_calls = 0

    def log_evaluation(self, dt, derivative=False):
        raise NotImplementedError
    
    def log_method_calls(self, dt):
        raise NotImplementedError

    def log_hardware(self):
        raise NotImplementedError
    
    def log_experiment(self, experiment):
        raise NotImplementedError
