from abc import ABCMeta, abstractmethod


class SamplingExperiment(Metaclass=ABCMeta):
    def __init__(self, method=None, location=None, overwrite=False):
        self.method = method
        self.logger = SamplingLogger(location, overwrite)
    
    def __call__(self, function, path):
        # Store testfunction
        self.function = function
        # Initialise method and get first queries
        self.method.initialise()
        # Perform sampling as long as procedure is not finished
        while not self.method.is_finished():
            pass

    def evaluate(self, x, derivative):
        raise NotImplementedError

    def analyse(self):
        raise NotImplementedError


class SamplingLogger:
    def __init__(self, path, overwrite=False):
        # TODO: Check if path is valid
        self.path = path
    
    def check_path(self, path):
        raise NotImplementedError
    
    def make_path(self, path):
        raise NotImplementedError

    def log_evaluation(self, dt, derivative=False):
        raise NotImplementedError

    def log_hardware(self):
        raise NotImplementedError
    
    def log_experiment(self, experiment):
        raise NotImplementedError
    
    def log_function(self, function):
        raise NotImplementedError


class SamplingResults:
    def __init__(self, path):
        raise NotImplementedError

    def load_results(self, path):
        raise NotImplementedError
    
    # TODO: add plot functions