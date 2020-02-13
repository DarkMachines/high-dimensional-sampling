# Creating a procedure

The high-dimensional-sampling (hds) package allows users to define sampling 
procedures and test them on a wide variety of test functions in a neatly 
automated an logged manner. Both the experimental framework and the test 
functions are defined within the package, the only thing the user has to do is 
define the procedure itself.

## Choosing the type of sampling

Within the hds project two different sampling tasks have been defined:

- **Posterior sampling**: taking samples from a distribution of which the
probability density (i.e. the function value) is only accessible through these
samples.
- **Optimisation**: Finding the optimum of an unknown function.

Implementing a procedure for either of these two can be done by creating a class
that inherits from the `procedures.Procedure` class:

    import high_dimensional_sampling as hds
    
    class MyProcedure(hds.Procedure):
        ...

This class should implement at least the following four methods:

### `__init__(self)`
This is Initialisation method in which anything can be done. At the very least
it should define the `self.store_parameters` property as a list, containing
the names of the parameters of the Procedure. These parameters will then 
automatically be stored when an experiment is run. If no procedure parameters
exist, `self.store_parameters` should be an empty list.

Your implementation of this initialisation method is allowed to include extra
input arguments, but all these input arguments should have default values
implemented, i.e.

    __init__(self, a=1, b=3.14, c=4)

As this method is an intialisation of the object, no parameters should be
returned from it.

### `__call__(self, function)`
The call method queries the class for the sampling of (a) new data point(s). It
takes a TestFunction instance as input and should return the sampled data
point(s) `x` and the labels for this(these) data point(s) `y`. The labels can
be obtained by [calling the test function](02_using_testfunctions.md). `x` 
should be a numpy array of shape `(nDatapoints, nVariables)`, whereas `y` 
should be a numpy array of shape `(nDatapoints, nOutputVariables)`. Note that 
`nOutputVariables` will often be 1.

Although it is not required, it is advisable to implement the `__call__` method
in such a way that it returns a single data point at a time. Whether or not
this is possible depends of course on the implemented procedure, but following
this guideline makes sure that the experiment is stopped as soon as possible
and no further (possibly expensive) iterations are run. This makes comparison
of different Procedures easier.

### `is_finished(self)`
This method checks if the Procedure is finished and the experiment can be
stopped. It should return a boolean, which is `True` is the experiment can be
stopped, or `False` if it should continue. If `True`, this overrides the
`finish_line` condition set in [the Experiment](04_running_an_experiment.md).

### `reset(self)`
This method should reset all internal variables concerning previous runs. For
instance: if your implementation keeps track of the number of points sampled
in a property called `n_sampled`, the `reset()` method should set `n_sampled`
to 0. It is called every time a new testfunction is provided to the Procedure.

Please do note that it should *not* reset possible configuration parameters
of the Procedure.

## Example scripts

In the [examples](../examples) folder two example procedures are provided: one
for [rejection sampling](../examples/rejection_sampling.py), a procedure for
posterior sampling, the other for 
[random optimisation](../examples/random_optimisation.py), a procedure for 
optimum finding.
