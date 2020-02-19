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


def test_hiddenfunctions_all():
    function = func.HiddenFunction1()
    x = np.random.rand(10, 2)
    assert function(x).shape[0] == len(x)
    function = func.HiddenFunction2()
    x = np.random.rand(7, 4)
    assert function(x).shape[0] == len(x)
    function = func.HiddenFunction3()
    x = np.random.rand(99, 6)
    assert function(x).shape[0] == len(x)
    function = func.HiddenFunction4()
    x = np.random.rand(1, 16)
    assert function(x).shape[0] == len(x)
