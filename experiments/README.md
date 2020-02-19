# How to perform experiments with the HDS framework

## Creating a procedure

To run an experiment, you need a procedure that performs an optimisation task. A range of procedures is already implemented in the `high_dimensional_sampling.optimisation` submodule. If you want to create your own procedure, please have a look at the [manual](https://github.com/DarkMachines/high-dimensional-sampling/blob/master/manual/01_creating_a_procedure.md). If you run into problems, don't hesitate to [open an issue](https://github.com/DarkMachines/high-dimensional-sampling/issues/new) in the GitHub project.

We aim to have all tested procedures included in the package. When your procedure is finished, have a look at the[project wiki](https://github.com/DarkMachines/high-dimensional-sampling/wiki/Contributing-your-algorithm-to-the-package) for more information on how to contribute your work to the package.

In the remainder of this README we will assume you have a working procedure implemented.

## Creating the experiment code

In the [experiments folder](https://github.com/DarkMachines/high-dimensional-sampling/tree/master/experiments) of the repository we have included a template script that you can use to test your procedure. To make this code work, you need to:

1. Set values for the configuration of the experiment script
2. Initialise your procedure

The template script guides you as to where you should put all relevant variables and values. Explanation for each of the configuration parameters is included in the template script as well.

After configuring the script you can run the code. Your procedure will then be tested against all `TestFunction`s and `HiddenFunction`s that you requested in the configuration of the script. The analytics of the experiments will be stored in location defined by the `RESULTS_FOLDER` variable in the experiment script.

## Performing experiments

Having run the experiment script, check the `<RESULTS_FOLDER>/<function name>/experiment.yaml` file for each of the for you interesting test functions. This file defines the best found value during optimisation, as well as the number of function evaluations and the configuration of the procedure and test function. Use this information to find the best function value whilst minimising the number of likelihood evaluations.

## Documenting your results

After the optimisation of your procedure, document your code by creating a pull request for it to the `experiments` folder in the repository. Information on how to create such a pull request can be found in the [project wiki](https://github.com/DarkMachines/high-dimensional-sampling/wiki/Contributing-your-algorithm-to-the-package). In this pull request you include:

- The code that you created from the template script. This script should define the best performing procedure parameters that you found in the previous step. Give this script the name of the procedure that you tested.
- The `RESULTS_FOLDER` with the results for all `TestFunction`s and `HiddenFunction`s for the best found procedure parameters. Let this be a clean folder, with only one subfolder for each of the functions. Give this folder the name of the procedure that you tested.

The documentation of the results is done in the project overleaf. Conventions for this will be put in the overleaf itself. The URL for this document can be given on request.
