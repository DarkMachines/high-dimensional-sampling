import high_dimensional_sampling as hds
from high_dimensional_sampling import results
import numpy as np

### PS stuff ###
import pyscannerbit.scan as sb

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

new_scans = True
###


class HDS_PS_INTERFACE(hds.Procedure):
    def __init__(self):
        self.store_parameters = []
        self.reset()
    
    def __call__(self, function):
        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.
        ranges = function.get_ranges(0.01)
        ranges = np.array(ranges).tolist()
        
        simple = function.get_simple_interface()
        # Can this be done without specifiying dimensionality? HDS and PS are looking for max/min respectively.
        neg_f = lambda x,y: -simple(x,y)

        dimensions = len(ranges)

        myscan = sb.Scan(neg_f, bounds = ranges, prior_types=["flat"]*dimensions, scanner=s, settings=settings)
        if new_scans:
            print("Running scan with {}".format(s))
            myscan.scan()
        else:
            print("Retrieving results from previous {} scan".format(s)) 
        results_ps = myscan.get_hdf5()
        names = results_ps.get_param_names()
           
        no_samples = len(results_ps.get_params(names[0])[0])
        #no_parameters = len(names)

        # Create array for sampled parameters
        x = np.zeros((no_samples,dimensions))
        i = 0
        for name in names:
            x[:, i] = results_ps.get_params(name)[0]
            i = i + 1
            # Print out best values for testing
            print(results_ps.get_best_fit(name))
        
        # No way to get sampled function values from PyScannerbit, so recalculate
        y = function(x) 
     
        return (x, y.reshape(-1, 1))

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return True

    def check_testfunction(self, function):
        return True



procedure = HDS_PS_INTERFACE( )
experiment = hds.OptimisationExperiment(procedure, '/home/zac/BayesianOptimisation/my_dm_clone/high-dimensional-sampling/pyscannerbit_interface_tests/results')
feeder = hds.functions.FunctionFeeder()
#feeder.load_function('Rastrigin', {'dimensionality': 2})
feeder.load_function('Beale')

### More PS stuff ###
# Try defining ps variables outside of class declaration
from collections import defaultdict
def rec_dd():
    return defaultdict(rec_dd)
settings = rec_dd()

# Settings for quick and dirty scans. Won't do very well, because the test function is
# actually rather tough!
scan_pars = settings["Scanner"]["scanners"]
scan_pars["multinest"] = {"tol": 0.5, "nlive": 100} 
scan_pars["polychord"] = {"tol": 1.0, "nlive": 20} 
scan_pars["diver"]     = {"convthresh": 1e-2, "NP": 300} 
scan_pars["twalk"]     = {"sqrtR": 1.05}
scan_pars["random"]    = {"point_number": 10000}
scan_pars["toy_mcmc"]  = {"point_number": 10} # Acceptance ratio is really bad with this scanner, so don't ask for much

scanners = ["diver"]

for s in scanners:
    for function in feeder:
        experiment.run(function, finish_line=1000)

df = results.make_dataframe({'simple': '/home/zac/BayesianOptimisation/my_dm_clone/high-dimensional-sampling/pyscannerbit_interface_tests/results'})
