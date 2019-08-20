# Selecting and Looping over `TestFunction`s

Running an experiment boils down to running an
[implemented procedure](01_creating_a_procedure.md) over multiple
`TestFunction`s. This can be rather boring to code, so a container class 
`functions.FunctionFeeder` has been implemented that allows you to select 
groups of testunctions and to loop over these functions for your experiment.

## Selecting individual testfunctions
The `FunctionFeeder` container class is part of the `functions` module:

    import high_dimensional_sampling as hds
    feeder = hds.functions.FunctionFeeder()

The `FunctionFeeder` can load individual functions by function name. For 
example

    feeder.load_function('Cosine')

This adds the Cosine test function to the feeder container. Some testfunctions
have configuration parameters, which can be set through an additional input
argument to the `load_function` method. For example:

    feeder.load_function('Rosenbrock', {'dimensionality': 3})

## Selecting multiple testfunctions at the same time

All [implemented testfunctions](02_using_testfunctions.md) are part of 
function groups. You can select all functions in a group by loading the group.
To load all functions with a implemented derivative for example, you can run

    feeder.load_function_group('with_derivative')

It is also possible to select intersections of multiple groups. For instance:
if you want to select all functions with an implemented derivative *that are
meant for optimisation problems*, the following code can be used:

    feeder.load_function_group(['with_derivative','optimisation'])

A second argument can be added to this function to add configuration to the
loaded testfunctions.

    config = {
        'Rosenbrock': {'dimensionality': 3},
        'Rastrigin': {'dimensionality': 6}
    }
    feeder.load_function_group('optimisation', config)

### Implemented groups
In the table below you can find an overview of all implemented groups and
which functions are part of these groups.

| Function | `optimisation `| `posterior` | `with_derivative` | `no_derivative` | `bounded` | `unbounded` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Rastrigin          | x |   | x |   | x |   |
| Rosenbrock         | x |   |   | x |   | x |
| Beale              | x |   |   | x | x |   |
| Booth              | x |   |   | x | x |   |
| BukinNmbr6         | x |   |   | x | x |   |
| Matyas             | x |   |   | x | x |   |
| LeviNmbr13         | x |   |   | x | x |   |
| Himmelblau         | x |   |   | x | x |   |
| ThreeHumpCamel     | x |   |   | x | x |   |
| Sphere             | x |   | x |   |   | x |
| Ackley             | x |   |   | x | x |   |
| Easom              | x |   |   | x | x |   |
| Cosine             |   | x | x |   | x |   |
| Block              |   | x |   | x |   | x |
| Bessel             |   | x | x |   | x |   |
| ModifiedBessel     |   | x | x |   | x |   |
| Eggbox             |   | x |   | x | x |   |
| MultivariateNormal |   | x |   | x | x |   |
| GaussianShells     |   | x |   | x | x |   |
| Linear             | x | x |   | x | x |   |
| BreitWigner        |   | x | x |   | x |   |
| Reciprocal         |   | x | x |   | x |   |

## Adding testfunctions by hand
Although the `FunctionFeeder` takes care of the correct initialisation and
configuration of testfunctions, if you feel more comfortable with adding
functions that you initialised yourself, you can do this as well. In that
case, use the `add_function` method:

    f = hds.functions.Cosine()
    feeder.add_function(f)

## Looping over testfunctions
After having selected and added all functions you want to loop over, you
can simply loop over these functions with a `for`-loop. If you want to print
all the function names, you can use the following code for example:

    for function in feeder:
        print(type(function).__name__)