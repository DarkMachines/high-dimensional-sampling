import high_dimensional_sampling as hds
import numpy as np
from turbo import TurboM
import sys


class BO_Turbo(hds.Procedure):
    def __init__(self,
                 max_evals=1000,
                 trust_regions=5,
                 max_cholesky_size=2000,
                 n_training_steps=50,
                 batch_size=10):
        self.max_evals = max_evals
        self.trust_regions = trust_regions
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.store_parameters = []
        self.batch_size = batch_size
        self.max_int = sys.maxsize
        self.min_int = -sys.maxsize - 1

    def __call__(self, function):
        ranges = function.get_ranges()
        dim = function.get_dimensionality()
        self.bounds = np.array(ranges)

        # Bounding the function for TuRBO optimization.
        for i in range(len(self.bounds)):
            for j in range(len(self.bounds[i])):
                if self.bounds[i][j] == float('inf'):
                    self.bounds[i][j] = self.max_int
                elif self.bounds[i][j] == float('-inf'):
                    self.bounds[i][j] = self.min_int

        if function.inverted:

            def func(function):
                return -function

            function = func

        turbo_m = TurboM(
            f=function,  # Handle to objective function
            lb=np.array(self.bounds[:, 0]),  # Numpy array specifying
                                             # lower bounds
            ub=np.array(self.bounds[:, 1]),  # Numpy array specifying upper
                                             # bounds
            n_init=dim,  # Number of initial bounds from an Symmetric Latin
                         # hypercube design
            max_evals=self.max_evals,  # Maximum number of evaluations
            n_trust_regions=self.trust_regions,  # Number of trust regions
            batch_size=self.batch_size,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP
                           # kernel
            max_cholesky_size=self.max_cholesky_size,  # When we switch
                                                       # from Cholesky to
                                                       # Lanczos
            n_training_steps=self.n_training_steps,  # Number of steps of
                                                     # ADAM to learn the
                                                     # hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )

        turbo_m.optimize()
        X = turbo_m.X  # Evaluated points
        fX = turbo_m.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]
        return x_best, f_best

    def is_finished(self):
        return False

    def check_testfunction(self, function):
        return True

    def reset(self):
        pass


procedure = BO_Turbo(max_evals=100)  # Default
experiment = hds.OptimisationExperiment(procedure, './hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function_group(['optimisation', 'bounded'])

for function in feeder:
    experiment.run(function, finish_line=1000)  # 1000
