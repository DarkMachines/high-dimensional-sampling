import os
import numpy as np
import high_dimensional_sampling.functions as func


def test_hiddenfunctions_general():
    function = func.HiddenFunction1()
    assert function.packageloc is None
    assumed_loc = 'hidden_functions{}test_func_1.bin'.format(os.sep)
    assert function.funcloc == assumed_loc
    assert function.ranges == [[-30.0, 30.0], [-30.0, 30.0]]
    x = np.random.rand(10, 2)
    print(function(x))
    assert function(x).shape[0] == len(x)
