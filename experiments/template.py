import os

import high_dimensional_sampling as hds
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import optimisation


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
#    func.HiddenFunction1(),
#    func.HiddenFunction2(),
#    func.HiddenFunction3(),
#    func.HiddenFunction4()
]
# The RUN_TEST_FUNCTIONS boolean indicates if the procedure should be tested
# against the all available test functions that have a known functional form.
RUN_TEST_FUNCTIONS = True

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
MAXIMUM_NUMBER_OF_SAMPLES = 1000

""" Output """
# Folder in which the results of the experiment will be stored. If the folder
# does not exist, it will be created.
RESULTS_FOLDER = "./hds"


""" ===========================================================================
        Procedure initialisation
=========================================================================== """

# Initialise the procedure that you want to use in your experiment. The
# object that you create should be stored in the `procedure` variable.
procedure = optimisation.RandomOptimisation(n_initial=5)


""" ===========================================================================
        Initialisation of experiment and functions
=========================================================================== """
# This code initialises the experiment by providing it with your procedure
# of choice and the folder to which results should be stored. No futher changes
# to this part of the code are necessary

# Initialise experiment
experiment = hds.OptimisationExperiment(procedure, RESULTS_FOLDER)

# Create function feeder
feeder = hds.functions.FunctionFeeder()
# Add test functions with known functional forms if this was requested through
# the RUN_TEST_FUNCTIONS configuration parameter
if RUN_TEST_FUNCTIONS:
    if PROCEDURE_REQUIRES_BOUNDED_FUNCTIONS:
        feeder.load_function_group(['optimisation', 'bounded'])
    else:
        feeder.load_function_group(['optimisation'])
# Add provided HiddenFunctions
for h in RUN_HIDDEN_FUNCTIONS_LIST:
    if isinstance(h, func.HiddenFunction):
        feeder.add_function(h)


""" ===========================================================================
        Performing the experiment on all test functions
=========================================================================== """

for function in feeder:
    # Invert function if requested
    if INVERSE_OPTIMISATION:
        function.invert()
    # Wrap function if requested
    if FUNCTION_WRAPPER is not None and hasattr(FUNCTION_WRAPPER, '__call__'):
        function = FUNCTION_WRAPPER(function)
    # Perform experiment with function
    experiment.run(function, finish_line=int(MAXIMUM_NUMBER_OF_SAMPLES))
