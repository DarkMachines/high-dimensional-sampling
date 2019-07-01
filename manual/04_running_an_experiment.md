# Running an experiment
As explained in [Creating a method](01_creating_a_method) the 
high_dimensional_sampling package is meant to investigate two distinct types
of sampling experiments: 

- **Posterior sampling**: taking samples from a distribution of which the
probability density (i.e. the function value) is only accessible through these
samples.
- **Optimisation**: Finding the optimum of an unknown function.

Each of these goals has its own associated Experiment class with it.

## Use the correct Experiment classes
The `experiments` module implements the classes for the experiment types. When
you want to run an optimisation experiment, you should use the 
`experiments.OptimisationExperiment` class (or the 
`experiments.OptimizationExperiment` class, which is totally equivalent). For
posterior sampling experiments the `experiments.PosteriorSamplingExperiment`
class should be used.

If you inspect the code you can see that there are no differences between these
two (three) classes. For the moment this is true, but in order to allow for
compatibility with future versions of the high_dimensional_sampling package
you are encouraged to use the correct experiment class now anyway.

## Running an experiment
Running an experiment boils down to the following steps:

1. Define a method to test
2. Define which testfunctions to test the method on
3. Determine the location to which results should be written
4. Loop over the selected testfunctions
5. Run the experiment on each of the functions

Given that there exists an implementation for the `MyMethod` class that is in
accordance with [Creating a method](01_creating_a_method.md), the following
code would test this method:

    import high_dimensional_sampling as hds

    method = MyMethod()

    feeder = hds.functions.FunctionFeeder()
    feeder.add_function_group('optimisation')

    experiment = hds.OptimisationExperiment(method, '/home/bstienen/log')
    for function in feeder:
        experiment.run(function, finish_line=1000, log_data=True)

Note that we provided a couple of arguments in the `OptimisationExperiment`
instance. The `'/home/bstienen/log'` argument defined the location to which
results should be written. This includes a basic benchmark of the computer
on which the experiment is performed. For each function the number of
function calls and the number of calls to the method is stored. If the
`log_data` argument in the `run()` method is set to True (default), also the
sampled data is logged.

The `finish_line` argument defines after how many taken samples the experiment
should be stopped. This is a *hard* stop: no matter the status of the sampling,
the experiment with the current testfunction is stopped. Note that although
the experiment can never be run longer than set through this argument, it
can stop earlier, if the Method's `is_finished()` method has returned `True`
before the finish line was reached.

## Example scripts

In the [examples](../examples) folder two example methods and experiments are 
provided: one for [rejection sampling](../examples/rejection_sampling.py), a 
method for posterior sampling, the other for 
[random optimisation](../examples/random_optimisation.py), a method for 
optimum finding.