# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

## Added

* [TuRBO](https://github.com/uber-research/TuRBO) is now implemented as a
  Procedure in the `optimisation` module. The code, and an example use, can be
  found in the `examples` folder as well.
* Additional tests on the shape of input data for test functions and on the
  shape of output data from procedures. This change was made to accomodate the
  usage of the TuRBO package.
* Argument `verbose` to Experiment initialisation method. This method controls
  the output of intermediate results (#samples for the #procedure_calls). See
  documentation of __init__ of the Experiment class for more information.
* The dimensionality of hidden `TestFunction`s can now be changed with the
  `dimensionality` argument at initialisation of the class.
* The ParticleFilter now implements a callback method that allows for the
  inclusion of a callback function in each sampling iteration (except for
  the first one). See the documentation on the wiki and in the docstring for
  more information.
* A `MSSM7` function has been added to the `functions` module. This function
  calls a trained neural network to evaluate the likelihood in a 12-dimensional
  space (which includes 5 nuisance parameters). The algorithm was trained on 
  data from the Gambit collaboration.

### Changed

* As `pygmo` cannot be installed through the pip installer, it has been
  removed from installation requirements in the `setup.py` file. This will only
  yield error messages when the `pygmo` package is actually requested by the
  implemented pygmo Procedure.
* The `weighing_deterministic_linear` function in the particle filter's linear
  function was based on sample order, not on function value. This has been
  changed.
* To accomodate the implementation of callbacks in the particle filter, the
  selector methods don't select data directly, but instead output `(indices,
  samples, values)`, where `indices` are the indices of the samples and values
  selected.
* Changed particle filter implementation to the one in
  https://github.com/bstienen/particlefilter.
* Plotting style now puts the grid at the lowest z-order, such that e.g.
  scatter markers are fully visible.

### Fixed

* Some print commands that were left from a debugging era are now removed.
* When using the `invert` method on a wrapped function, the original function
  was also inverted. This was solved by having the `get_simple_interface()`
  and `get_simple_interface_with_scan` methods use copies of the original
  function.
* The weighing_stochastic_linear function in the particle filter optimisation
  method could occasionally raise errors related to `nan` probabilities. This
  is now fixed.
* The gaussian constructors needed for the particle filter could return negative
  results. As the output represents standard deviations, this is unexpected
  behaviour. The absolute value of the standard deviation is now returned
  instead.
* The particle filter did not implement a way to keep the best points of the
  previous iteration for the current one. This is now implemented through the
  `survival_rate` argument (default=0.2).

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
