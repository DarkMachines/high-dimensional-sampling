# How to run:
# 1) sudo apt-get install python-pip
# 2) pip install gpyopt

import high_dimensional_sampling as hds
import numpy as np

import GPyOpt


class Gpyopt(hds.Procedure):
    def __init__(self,
                 initial_design_numdata=5,
                 aquisition_type='EI',
                 exact_feval=True,
                 de_duplication=True,
                 num_cores=2,
                 max_iter=1000,
                 max_time=300,
                 eps=1e-6):
        self.store_parameters = ['initial_design_numdata',
                                 'aquisition_type',
                                 'exact_feval',
                                 'de_duplication',
                                 'num_cores',
                                 'max_iter',
                                 'max_time',
                                 'eps']
        self.initial_design_numdata = initial_design_numdata
        self.aquisition_type = aquisition_type
        self.exact_feval = exact_feval
        self.de_duplication = de_duplication
        self.num_cores = num_cores
        self.max_iter = max_iter
        self.max_time = max_time
        self.eps = eps
        self.reset()

    def __call__(self, function):
        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        ranges = np.array(ranges).tolist()

        dimensions = function.get_dimensionality()

        mixed_domain = []
        for dim in range(0, dimensions):
            mixed_domain.append({'name': 'var'+str(dim+1),
                                 'type': 'continuous',
                                 'domain': (ranges[dim][0], ranges[dim][1]),
                                 'dimensionality': 1})

        myBopt = GPyOpt.methods.BayesianOptimization(
                    f=function,
                    domain=mixed_domain,
                    initial_design_numdata=self.initial_design_numdata,
                    acquisition_type=self.aquisition_type,
                    exact_feval=self.exact_feval,
                    de_duplication=self.de_duplication,
                    num_cores=self.num_cores)

        max_iter = self.max_iter       # maximum number of iterations
        max_time = self.max_time       # maximum allowed time
        eps = self.eps  # tolerance, max dist between consecutive evaluations.

        myBopt.run_optimization(max_iter, max_time, eps)

        x = myBopt.get_evaluations()[0]
        y = myBopt.get_evaluations()[1]

        # np.round(myBopt.X,2)
        myBopt.save_report("gpyopt_report.txt")
        # myBopt.plot_acquisition()
        # myBopt.plot_convergence()

        return (x, y)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return True

    def check_testfunction(self, function):
        return True
