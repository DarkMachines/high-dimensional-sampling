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

""" Procedure configuration """
# The MAXIMUM_NUMBER_OF_SAMPLES configuration defines how many samples are to
# be sampled. If after any iteration the number of samples exceeds this number,
# the optimisation procedure is stopped.
MAXIMUM_NUMBER_OF_SAMPLES = 10000

convthresh_values = [1e-2]
np_values = [2000]


# Create function feeder
feeder = hds.functions.FunctionFeeder()
mssm7 = func.MSSM7()
feeder.add_function(mssm7)
# feeder.add_function(func.HiddenFunction1(int(2)))


for numpoints in np_values:
    for convthresh in convthresh_values:

        RESULTS_FOLDER = "/home/user/high-dimensional-sampling/" \
            "mssm7_functest_diver_"+str(convthresh)+"_"+str(numpoints)

        procedure = optimisation.PyScannerBit(scanner="diver",
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
                                              pso_np=400,
                                              output_path=RESULTS_FOLDER)

        experiment = hds.OptimisationExperiment(procedure, RESULTS_FOLDER)

        # Perform experiment with function
        for function in feeder:
            # if MPI.COMM_WORLD.Get_rank() == 0:
            experiment.run(function,
                           finish_line=int(MAXIMUM_NUMBER_OF_SAMPLES))
