import numpy as np
import pandas as pd
import pytest
import high_dimensional_sampling.functions as func


# An instances of the TestFunction class with simple behaviour
class TmpFunction(func.TestFunction):
    def __init__(self, **kwargs):
        self.ranges = [[0, 1], [0, 1]]
        super(TmpFunction, self).__init__(**kwargs)

    def _evaluate(self, x):
        return x

    def _derivative(self, x):
        raise func.NoDerivativeError()


class TmpFunction2(func.TestFunction):
    def __init__(self, dimensionality=3, **kwargs):
        self.ranges = self.construct_ranges(dimensionality, -10, 10)
        super(TmpFunction2, self).__init__(**kwargs)

    def _evaluate(self, x):
        return np.sum(x**2, axis=1)

    def _derivative(self, x):
        return 2*x


class TmpFunction3(func.TestFunction):
    def __init__(self, dimensionality=3, **kwargs):
        super(TmpFunction3, self).__init__(**kwargs)

    def _evaluate(self, x):
        return np.sum(x**2, axis=1)

    def _derivative(self, x):
        return 2*x


def test_function_dimensionality():
    tmp = TmpFunction()
    tmp2 = TmpFunction2()
    # Test if function dimensionality is correctly identified
    assert tmp.get_dimensionality() == 2
    assert tmp2.get_dimensionality() == 3
    # Hand pick dimensionality
    tmp2.ranges = [[0, 1], [0, 1], [0, 1], [0, 1]]
    assert tmp2.get_dimensionality() == 4
    tmp2.ranges = tmp2.construct_ranges(10, -10, 10)
    assert tmp2.get_dimensionality() == 10
    tmp2.ranges = tmp2.construct_ranges(1, -10, 10)
    assert tmp2.get_dimensionality() == 1


def test_differentiability():
    tmp = TmpFunction()
    tmp2 = TmpFunction2()
    # Test if function differentiability is correctly identified
    with pytest.raises(Exception):
        tmp._derivative(np.ones((10, 2)))
    assert not tmp.is_differentiable()
    assert tmp2.is_differentiable()


def test_boundedness():
    tmp = TmpFunction()
    # Test if function boundedness is correctly identified
    assert tmp.is_bounded()
    tmp.ranges = [[-np.inf, np.inf]]
    assert not tmp.is_bounded()
    tmp.ranges = [[0, np.inf]]
    assert not tmp.is_bounded()
    tmp.ranges = [[-np.inf, 1]]
    assert not tmp.is_bounded()


def test_input_conversion():
    tmp = TmpFunction()
    # Test if input into function is converted to numpy array
    x = np.random.rand(100, 2)
    y = tmp.to_numpy_array(x)
    assert isinstance(y, np.ndarray)
    y = tmp.to_numpy_array(x.tolist())
    assert isinstance(y, np.ndarray)
    y = tmp.to_numpy_array(pd.DataFrame(x))
    assert isinstance(y, np.ndarray)
    with pytest.raises(Exception):
        tmp.to_numpy_array("not_allowed_input_format")


def test_dimensionality_check():
    tmp = TmpFunction()
    tmp.ranges = np.array(tmp.ranges)
    # Test if input data is correctly checked for dimensionality
    x = np.random.rand(200, 2)
    tmp.check_dimensionality(x.shape)  # assert is implicit
    x = np.random.rand(100, 2)
    tmp.check_dimensionality(x.shape)  # assert is implicit
    with pytest.raises(Exception):
        tmp.check_dimensionality(x.reshape(50, 4))


def test_range_check():
    tmp = TmpFunction()
    tmp.ranges = np.array(tmp.ranges)
    # Check if ranges of input data are checked
    # First checks with an epislon of 0
    x = np.random.rand(100, 2)
    tmp.check_ranges(x, epsilon=0)  # assert is implicit
    with pytest.raises(Exception):
        tmp.check_ranges(x*2, epsilon=0)
    # Now checks with a non-zero epsilon
    tmp.check_ranges(x*0.8+0.1, epsilon=0.1)
    with pytest.raises(Exception):
        tmp.check_ranges(x, epsilon=0.1)
    with pytest.raises(Exception):
        tmp.check_ranges(x*0.9, epsilon=0.1)
    with pytest.raises(Exception):
        tmp.check_ranges(x*0.9+0.1, epsilon=0.1)


def test_configuration_check():
    tmp = TmpFunction()
    # Check if ranges are defined and contert to numpy array if not already
    # Default list ranges
    tmp.ranges = [[0, 1], [0, 1]]
    assert isinstance(tmp.ranges, list)
    tmp.check_configuration()
    assert isinstance(tmp.ranges, np.ndarray)
    # Dynamically constructed ranges
    tmp.ranges = tmp.construct_ranges(2, 0, 1)
    assert isinstance(tmp.ranges, list)
    tmp.check_configuration()
    assert isinstance(tmp.ranges, np.ndarray)
    # Raise exception when no range is defined
    del(tmp.ranges)
    with pytest.raises(Exception):
        tmp.check_configuration()


def test_run_testfunction():
    tmp = TmpFunction()
    tmp2 = TmpFunction2()
    # Check if input type is not important (list, pandas, numpy)
    x = np.random.rand(100, 2)
    tmp(x)  # assert is implicit
    tmp(x.tolist())  # assert is implicit
    tmp(pd.DataFrame(x))  # assert is implicit
    # Check if inversion works
    z = tmp(x)
    tmp.invert()
    assert np.array_equal(z, -1*tmp(x))
    tmp.invert(False)
    assert np.array_equal(z, tmp(x))
    # Check if dimensionality check is performed
    with pytest.raises(Exception):
        tmp(x.reshape(50, 4))
    # Check if ranges check is performed
    tmp(x*0.9)  # assert is implicit
    tmp(x*0.8+0.1, epsilon=0.1)  # assert is implicit
    with pytest.raises(Exception):
        tmp(x*2)
    with pytest.raises(Exception):
        tmp(x*0.9, epsilon=0.1)
    # Check if derivative can be called
    x = np.random.rand(100, 3)
    tmp2(x, derivative=True)  # assert is implicit
    with pytest.raises(Exception):
        tmp(x, derivative=True)


def test_internal_counter():
    tmp = TmpFunction()
    tmp2 = TmpFunction2()
    # Check if number of calls is 0 at initialisation
    assert len(tmp.counter) == 0
    assert tmp.count_calls() == (0, 0)
    assert tmp.count_calls("normal") == (0, 0)
    assert tmp.count_calls("derivative") == (0, 0)
    with pytest.raises(Exception):
        tmp.count_calls("not_allowed_mode")
    # Check if number of calls is increased
    x = np.random.rand(100, 2)
    tmp(x)
    assert len(tmp.counter) == 1
    assert tmp.count_calls() == (1, 100)
    assert tmp.count_calls("normal") == (1, 100)
    assert tmp.count_calls("derivative") == (0, 0)
    tmp(x[:36])
    assert len(tmp.counter) == 2
    assert tmp.count_calls() == (2, 136)
    assert tmp.count_calls("normal") == (2, 136)
    assert tmp.count_calls("derivative") == (0, 0)
    # Check if this also works with derivatives
    x = np.random.rand(100, 3)
    tmp2(x)
    assert len(tmp2.counter) == 1
    assert tmp2.count_calls() == (1, 100)
    assert tmp2.count_calls("normal") == (1, 100)
    assert tmp2.count_calls("derivative") == (0, 0)
    tmp2(x, derivative=True)
    assert len(tmp2.counter) == 2
    assert tmp2.count_calls() == (2, 200)
    assert tmp2.count_calls("normal") == (1, 100)
    assert tmp2.count_calls("derivative") == (1, 100)
    tmp2(x[:36], derivative=True)
    assert len(tmp2.counter) == 3
    assert tmp2.count_calls() == (3, 236)
    assert tmp2.count_calls("normal") == (1, 100)
    assert tmp2.count_calls("derivative") == (2, 136)
    # Check if reset works
    tmp2.reset()
    assert len(tmp2.counter) == 0
    assert tmp2.count_calls() == (0, 0)
    assert tmp2.count_calls("normal") == (0, 0)
    assert tmp2.count_calls("derivative") == (0, 0)


def test_rangeless_function():
    # Check if exception is raised when a rangeless function is initialised
    with pytest.raises(Exception):
        _ = TmpFunction3()


def test_simplefunctionwrapper():
    tmp = TmpFunction()
    # Test if initialisation is okay
    _ = func.SimpleFunctionWrapper(tmp)
    with pytest.raises(Exception):
        _ = func.SimpleFunctionWrapper("invalid_input")


def test_simplefunctionwrapper_call():
    tmp = TmpFunction2()
    wrapped = tmp.get_simple_interface()
    # Test if conversion to 2d array is correct
    x = np.random.rand(100, 3)
    z = wrapped._create_input_array([x[:, 0], x[:, 1], x[:, 2]])
    assert x.shape == z.shape
    assert np.sum(1.0*(x == z)) == 300
    # Check if inversion works
    z_evaluated = wrapped(x[:, 0], x[:, 1], x[:, 2])
    wrapped.invert()
    assert np.array_equal(z_evaluated, -1*wrapped(x[:, 0], x[:, 1], x[:, 2]))
    wrapped.invert(False)
    assert np.array_equal(z_evaluated, wrapped(x[:, 0], x[:, 1], x[:, 2]))
    # Test what happens when just numbers are provided
    z = wrapped._create_input_array([1, 2, 3])
    assert z.shape == (1, 3)
    # Check if exception is raised when input dimensionality is incorrect
    with pytest.raises(Exception):
        _ = wrapped(x[:, 0], x[:, 1])
    with pytest.raises(Exception):
        _ = wrapped(x[:, 0], x[:, 1], x[:, 2], x[:, 0])
    # Check if evaluation is same as unwrapped
    y1 = tmp(x)
    y2 = wrapped(x[:, 0], x[:, 1], x[:, 2])
    assert y1.shape == y2.shape
    assert np.sum(1.0*(y2 == y2)) == 100
    z1 = tmp(x, derivative=True)
    z2 = wrapped(x[:, 0], x[:, 1], x[:, 2], derivative=True)
    assert z1.shape == z2.shape
    assert np.sum(1.0*(z2 == z2)) == 300
    # Check if output of single point is a single value
    z = wrapped(1, 2, 3)
    print(z, type(z))
    assert isinstance(z, float)


def test_simplefunctionwrapper_propertycheckers():
    tmp = TmpFunction2()
    wrapped = tmp.get_simple_interface()
    # Make sure that property checkers of the SimpleFunctionWrapper get the
    # correct properties from the wrapped function
    assert wrapped.get_dimensionality() == tmp.get_dimensionality()
    assert wrapped.is_differentiable() == tmp.is_differentiable()
    assert wrapped.is_inverted() == tmp.inverted
    assert wrapped.is_bounded() == tmp.is_bounded()


def test_return_wrapper():
    # Check if a SampleFunctionWrapper is returned when asked
    tmp = TmpFunction()
    tmp_wrapped = tmp.get_simple_interface()
    assert isinstance(tmp_wrapped, func.SimpleFunctionWrapper)
    assert isinstance(tmp, type(tmp_wrapped.function))


def test_invertion():
    tmp = TmpFunction()
    tmp_wrapped = tmp.get_simple_interface()
    x = np.random.rand(1, 2)

    y_normal = tmp(x)
    y_wrapped_normal = tmp_wrapped(x[:, 0], x[:, 1])
    tmp_wrapped.invert()
    y_normal_inverted = tmp(x)
    y_wrapped_inverted = tmp_wrapped(x[:, 0], x[:, 1])

    assert np.array_equal(y_wrapped_normal, y_normal)
    assert np.array_equal(y_normal, -y_wrapped_inverted)
    assert np.array_equal(y_normal, y_normal_inverted)
