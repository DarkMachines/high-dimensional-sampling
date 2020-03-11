# Changelog

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

## Unreleased

## Added

### Changed

* As `pygmo` cannot be installed through the pip installer, it has been
  removed from installation requirements in the `setup.py` file. This will only
  yield error messages when the `pygmo` package is actually requested by the
  implemented pygmo Procedure.

### Fixed

## Version 0.2.0 (Monday February 17th, 2020)

### Added

* A submodule `results` that implements functionality to visualise or format
  the results of one or more experiments into neat little plots and tables.
  Accompanying this new submodule is a new example script.
* The Experiment class now is an abstract class with abstract methods that
  function as events. This allows users to more easily create Experiment
  classes that extend the functionality of the Experiment class.
* The `experiment.yaml` log now also contains entries on the experiment type
  (e.g. OptimisationExperiment) and the number of calls for the derivative
  of the TestFunction.
* The interface to TestFunctions was not working for all users. An interface
  class (SimpleFunctionWrapper) has been added that allows the user to provide
  the input parameters each as a separate argument. The interface can be
  activated by calling the `get_simple_interface` method of a TestFunction. An
  example using the different interfaces has been added
  (`testfunction_interfaces.py`).
* Add methods to the TestFunctions that check properties of the function. These
  include `is_differentiable`, `is_bounded` and `get_dimensionality`.
* Subpackage for algorithm implementations. RandomOptimisation and
  RejectionSampling (from the example scripts) have been added as examples of
  how to add implementations.
* Unit tests for the entire package are added.
* Testfunctions from precompiled binaries were added, so that the exact
  formula for them is unknown.
* Several optimisation procedures were added.

### Changed

* The examples now use relative path indications for storage of their results,
  so that the examples can be used out of the box without errors.
* The package indicator in setup.py now is more general, allowing for easier
  inclusion of submodules.
* Procedures should now implement a method called `check_testfunction` that
  returns a boolean, indicating if the provided test function can be used in
  the procedure.
* The ranges of the Block testfunction now go all the way to `-np.inf` and 
  `np.inf`, as is in accordance with the configuration of the function groups
  in the FunctionFeeder.

### Fixed

* An error was raised when GaussianShells was logged, as this function stores
  its parameters internally as numpy arrays. These don't translate well to
  .yaml files. This has been solved by forcing the storage of function 
  properties as lists if they are numpy arrays.
* Updated outdated docstrings at various locations in the package
* Fixed error causing testfunctions not to work with Pandas dataframes. They
  work with dataframes now, as was intended.
  
---

## Version 0.1.1 (Tuesday August 20th, 2019)

### Added

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

### Changed

* Increased the readability of the error message given when a test function is
  queried for its value outside of the box defined by its ranges parameter.
* The `utils.get_time` method now returns time in seconds

### Fixed

* Ackley, Easom and Sphere test functions returned data in an incorrect shape.
  This has been corrected.
* The GaussianShells test function mapped multi-point input to a single output
  value. This has been fixed.
