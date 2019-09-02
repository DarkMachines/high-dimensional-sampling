###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

Unreleased
**********

Added
-----
* The Experiment class how is an abstract class with abstract methods that
  function as events. This allows users to more easily create Experiment
  classes that extend the functionality of the Experiment class.
* The `experiment.yaml` log now also contains entries on the experiment type
  (e.g. OptimisationExperiment) and the number of calls for the derivative
  of the TestFunction.

Changed
-------
* The examples now use relative path indications for storage of their results,
  so that the examples can be used out of the box without errors.
* The package indicator in setup.py now is more general, allowing for easier
  inclusion of submodules.

Fixed
-----
* An error was raised when GaussianShells was logged, as this function stores
  its parameters internally as numpy arrays. These don't translate well to
  .yaml files. This has been solved by forcing the storage of function 
  properties as lists if they are numpy arrays.

Version 0.1.1 (Tuesday August 20th, 2019)
*****************************************

Added
-----
* A `get_ranges` method in the TestFunction class. It returns the ranges of the
  test function, taking a leeway parameter epsilon provided to the method into
  account: all minima will be raised by epsilon, all maxima will be reduced by
  it.
* The TestFunction `__call__` and `check_ranges` methods now also implement the
  epsilon argument (see first addition above).
* Users requested a smaller range for the Easom test function. Although the
  range [-100, 100] is normal for this function, to accommodate this wish the
  boundaries of this application box are now configurable at initialisation
  of the test function.
* The plot_testfunction example script has been splitted into two separate
  scripts: one for 1d and another for 2d test functions.
* When an experiment ends, the results (i.e. the elapsed time and the number
  of total function calls) are stored in the experiment.yaml file. Note that
  the logged time includes also the overhead introduced by the HDS package
  itself.
* The following test functions are added:
    * BreitWigner
    * Reciprocal

Changed
-------
* Increased the readability of the error message given when a test function is
  queried for its value outside of the box defined by its ranges parameter.
* The `utils.get_time` method now returns time in seconds

Fixed
-----
* Ackley, Easom and Sphere test functions returned data in an incorrect shape.
  This has been corrected.
* The GaussianShells test function mapped multi-point input to a single output
  value. This has been fixed.
