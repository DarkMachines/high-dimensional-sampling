import pytest
import numpy as np
import high_dimensional_sampling.functions as func

# An instance of the TestFunction class with simple behaviour
class TmpFunction(func.TestFunction):
    def __init__(self):
        self.ranges = [[0,1],[0,1]]
        super(TmpFunction, self).__init__()

    def _evaluate(self, x):
        return x

    def _derivative(self, x):
        raise func.NoDerivativeError()

def test_function_properties():
    tmp = TmpFunction()
    assert tmp.get_dimensionality() == 2
    assert tmp.is_bounded()
    tmp.ranges = [[0,1], [0,1], [0,1], [0,1]]
    assert tmp.get_dimensionality() == 4
    assert not tmp.is_differentiable()
    tmp.ranges = [[-np.inf, np.inf]]
    assert tmp.get_dimensionality() == 1
    assert not tmp.is_bounded()
    tmp.ranges = [[0, np.inf]]
    assert not tmp.is_bounded()
    tmp.ranges = [[-np.inf, 1]]
    assert not tmp.is_bounded()

  