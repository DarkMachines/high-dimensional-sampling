# How to run:
# 1) Clone pyscannerbit using:
#  git clone https://github.com/bjfar/pyscannerbit.git
# 2) Navigate to pyscannerbit directory (cd pyscannerbit)
# 3) Run "python setup.py sdist"
# 4) Navigate to 'sdist' folder and record name of tarball
# 5) Return to 'pyscannerbit' folder ("cd ..")
# 6) Run "pip install dist/pyscannerbit-X.X.X.tar.gz",
#  where X is the version number you noted in step 4
# 7) Navigate to 'tests' directory and run "python
#  test_all_scanners.py".
# 8) If successful, congrats. You have successfully installed
#  pyscannerbit. One possible error previously noted was that
#  yaml-cpp and pybind11 weren't automatically cloning along
#  with pyscannerbit, which can be fixed by manually cloning
#  them to the provided folders.

import high_dimensional_sampling as hds
from high_dimensional_sampling.optimisation import PyScannerBit

scanners = ["pso"]

for s in scanners:
    procedure = PyScannerBit(scanner=s,
                             multinest_tol=0.5,
                             multinest_nlive=100,
                             polychord_tol=1.0,
                             polychord_nlive=20,
                             diver_convthresh=1e-2,
                             diver_NP=300,
                             twalk_sqrtr=1.05,
                             random_point_number=10000,
                             toy_mcmc_point_number=10,
                             badass_points=1000,
                             badass_jumps=10,
                             pso_NP=400)
    experiment = hds.OptimisationExperiment(procedure, './hds')
    feeder = hds.functions.FunctionFeeder()
    feeder.load_function('Rastrigin', {'dimensionality': 3})
    for function in feeder:
        experiment.run(function, finish_line=1000)
