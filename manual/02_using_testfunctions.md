# Using `TestFunction`s
The `high_dimensional_sampling` package implements a variety of functions on
which sampling procedures can be tested. These are all classes with the 
`functions.TestFunction` class as their base. This base class automatically
keeps track of the number of function evaluations and performs checks on the
input data to make sure it has the correct number of dimensions. The 
`TestFunction` class is abstract and can not be instantiated itself.

## Evaluating the function('s derivative)
As testfunctions are implemented as classes, they can be evaluated after
initialisation. For example, if we want to evaluate the `Cosine` testfunction
at `x=2.65`, we would use the following script:

    import high_dimensional_sampling as hds

    f = hds.functions.Cosine()
    y = f(2.65)

Some testfunctions also have their derivative implemented. To get the
derivative at a specific location, simply add `True` as second argument to the
function call:

    y_prime = f(2.65, True)

If no derivative is implemented for the instantiated testfunction, a 
`functions.NoDerivativeError` is raised.

## Implemented functions
Currently the following functions are implemented in the `functions` module
of the package:

| Function | #dimensions | Derivative implemented | Definition  |
| ------------- |:-------------:|:-------------:| -----:|
| Rastrigin | (2) | Yes | https://en.wikipedia.org/wiki/Rastrigin_function |
| Rosenbrock | (2) | No | https://en.wikipedia.org/wiki/Rosenbrock_function |
| Beale | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Booth | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| BukinNmbr6 | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Matyas | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| LeviNmbr13 | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Himmelblau | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| ThreeHumpCamel | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Sphere | (3) | Yes |  https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Ackley | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Easom | 2 | No | https://en.wikipedia.org/wiki/Test_functions_for_optimization |
| Cosine | 1 | Yes | $$\cos( x )$$ |
| Block | (3) | No | $$a + b*\theta(\|x\|_0)$$ |
| Bessel | 1 | Yes | https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html |
| | | | https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j0.html |
| ModifiedBessel | 1 | Yes | https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html |
| |  | | https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0.html |
| Eggbox | 2 | No | https://arxiv.org/pdf/0809.3437.pdf |
| MultivariateNormal | (2) | No | https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html |
| GaussianShells | 2 | No | https://arxiv.org/pdf/0809.3437.pdf |
| Linear | (2) | No | $$\sum_i \| x_i \|$$ |
| BreitWigner | 1 | Yes | https://en.wikipedia.org/wiki/Relativistic_Breit–Wigner_distribution |
| Reciprocal | (2) | Yes | $$\prod_i x_i^{-1}$$ |

Dimensionalities provided between parentheses can be configured at
initialisation.

## Hidden TestFunctions
For all functions in the table above the function can in principle be known 
by the user performing the optimisation. This makes for a possible biassed
optimisation, as the user can tune the parameters of the optimisation
procedure. To counteract this, there are additionally four hidden functions
implemented, which get their evaluated values from precompiled binaries. This
makes their functional values unknown to the user.

The `HiddenFunction`s are implemented in the `functions` module and have the
following properties.

| Function | #dimensions | Range minimum | Range maximum |
| ------------- |:-------------:|:-------------:| -----:|
| HiddenFunction1 | 2 | -30 | 30 |
| HiddenFunction2 | 4 | -7 | 7 |
| HiddenFunction3 | 6 | 0 | 1 |
| HiddenFunction4 | 16 | -500 | 500 |

None have their derivative implemented and they don't need take any arguments 
at initialisation.

## Defining your own `TestFunction`s
You can define your own testfunctions by creating a class that derives from the
`functions.TestFunction` base class. Your new testfunction should implement
the following methods:

- **__init__(self, ...)**: initialisation method. Can take extra input in the
form of extra input arguments, but any such parameters should have defaults
defined.
- **_evaluate(self, x)**: method that evaluates the test function at coordinate
`x`. Be aware that `x` is a numpy array of shape `(nDatapoints, nVariables)`
and can contain multiple rows (i.e. data points). This method should return
the function values of your test function in the form of a numpy array of shape
`(nDatapoints, nOutputVariables)`.
- **_derivative(self, x)**: same as `_evaluate`, but now for the derivative of
the testfunction. If no derivative is implemented, this method should raise
a `functions.NoDerivativeError`.

## Looping over functions
Have a look at
[Using and Looping over TestFunctions](03_selecting_and_looping_over_testfunctions.md)
for more information on how to automatically select and loop over test
functions.
