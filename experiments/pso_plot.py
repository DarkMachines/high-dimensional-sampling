import os

import high_dimensional_sampling as hds
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import optimisation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as pylab
import sys

numdim = sys.argv[1]
print('Number of dimensions:',numdim)

""" ===========================================================================
        Configuration
=========================================================================== """

""" Functions to test the procedure with """
# Set this boolean flag to `True` if your procedure requires the test functions
# to have a bounded input range. Note that this only influences the *known*
# test functions, as the hidden test functions are all bounded
PROCEDURE_REQUIRES_BOUNDED_FUNCTIONS = False
# RUN_HIDDEN_FUNCTIONS_LIST is a list of *initialised* HiddenFunctions to test 
# the procedure against. If no HiddenFunction should be tested, provide an 
# empty list.
RUN_HIDDEN_FUNCTIONS_LIST = [
    func.HiddenFunction1(int(numdim)),
    func.HiddenFunction2(int(numdim)),
    func.HiddenFunction3(int(numdim)),
    func.HiddenFunction4(int(numdim))
]
# The RUN_TEST_FUNCTIONS boolean indicates if the procedure should be tested
# against the all available test functions that have a known functional form.
RUN_TEST_FUNCTIONS = False

""" Configuration of test and hidden functions """
# The HDS framework interprets optimisation as a minimalisation task. If your
# procedure sees optimisation as a maximisation task, set the following
# INVERSE_OPTIMISATION flag to `True`
INVERSE_OPTIMISATION = False
# The interface of the test functions might not match the interface that your
# procedure expects test functions to have. This is especially the case if the
# procedure is implemented in a third-party library. To solve this, the
# `TestFunction` instances can be wrapped such that the interface matches the
# required one. The FUNCTION_WRAPPER variable should in that case contain an
# *uninitialised* wrapper class. For instance:
#  
#     FUNCTION_WRAPPER = func.SimpleFunctionWrapper
#
# If you don't need your functions to be wrapped, FUNCTION_WRAPPER should be
# set to `None`.
FUNCTION_WRAPPER = None

""" Procedure configuration """
# The MAXIMUM_NUMBER_OF_SAMPLES configuration defines how many samples are to 
# be sampled. If after any iteration the number of samples exceeds this number,
# the optimisation procedure is stopped.
MAXIMUM_NUMBER_OF_SAMPLES = 10000

""" Output """
# Folder in which the results of the experiment will be stored. If the folder
# does not exist, it will be created.
#RESULTS_FOLDER = "/home/mwhi/high-dimensional-sampling/diver"

# Now we reach the part of the script where we will loop over Diver settings

convthresh_values = [1e-4,1e-3,1e-2,1e-1]
np_values = [2000,5000,10000,20000]

# Loop over np values, and make a plot for each case
# i.e. similar to top panel of Fig. 5 on page 22 of the ScannerBit paper


# Create function feeder
feeder = hds.functions.FunctionFeeder()
for h in RUN_HIDDEN_FUNCTIONS_LIST:
    if isinstance(h, func.HiddenFunction):
        feeder.add_function(h)

for numpoints in np_values:
        
    bestfit_logL_values = np.zeros(len(convthresh_values))
    count = 0
    for convthresh in convthresh_values:
        
        RESULTS_FOLDER = "/home/user/high-dimensional-sampling/" + numdim + "D_pso_"+str(convthresh)+"_"+str(numpoints)
        
        procedure = optimisation.PyScannerBit(scanner="pso",
                                              multinest_tol=0.5,
                                              multinest_nlive=100,
                                              polychord_tol=1.0,
                                              polychord_nlive=20,
                                              diver_convthresh=convthresh,
                                              diver_np=numpoints,
                                              twalk_sqrtr=1.05,
                                              random_point_number=10000,
                                              toy_mcmc_point_number=10,
                                              badass_points=1000,
                                              badass_jumps=10,
                                              pso_np=numpoints,
                                              pso_convthresh=convthresh,
                                              output_path=RESULTS_FOLDER)
        
        experiment = hds.OptimisationExperiment(procedure, RESULTS_FOLDER)
        
        # Perform experiment with function
        for function in feeder:
            #if MPI.COMM_WORLD.Get_rank() == 0:
            experiment.run(function, finish_line=int(MAXIMUM_NUMBER_OF_SAMPLES))
            
            
