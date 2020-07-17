import numpy as np
import high_dimensional_sampling.functions as func
import pytest


def test_testfunction_rastrigin():
    function = func.Rastrigin()
    minima = [[0, 0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    assert function.a == 10
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_rosenbrock():
    function = func.Rosenbrock()
    minima = [[1]*function.get_dimensionality()]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is False
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate error is raised when dimensionality < 2
    with pytest.raises(Exception):
        function = func.Rosenbrock(dimensionality=1)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_beale():
    function = func.Beale()
    minima = [[3.0, 0.5]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_booth():
    function = func.Booth()
    minima = [[1.0, 3.0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_bukinnmbr6():
    function = func.BukinNmbr6()
    minima = [[-10.0, 1.0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    print('bukinnmbr6', minima, y)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_matyas():
    function = func.Matyas()
    minima = [[0.0, 0.0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_levinmbr13():
    function = func.LeviNmbr13()
    minima = [[1.0, 1.0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_himmelblau():
    function = func.Himmelblau()
    minima = [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_threehumpcamel():
    function = func.ThreeHumpCamel()
    minima = [[0, 0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_sphere():
    function = func.Sphere()
    minima = [[0, 0, 0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 3
    # Validate function properties
    assert function.is_bounded() is False
    assert function.is_differentiable() is True
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x = x*20 - 10
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_ackley():
    function = func.Ackley()
    minima = [[0, 0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_easom():
    function = func.Easom()
    minima = [[np.pi, np.pi]]
    value_minima = -1
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    assert function.ranges[0][0] == -100
    assert function.ranges[0][1] == 100
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_cosine():
    function = func.Cosine()
    minima = [[-3*np.pi], [-np.pi], [np.pi], [3*np.pi]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 1
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_block():
    function = func.Block()
    # Validate the default configuration
    assert function.get_dimensionality() == 3
    assert function.block_size == 1
    assert function.block_value == 1
    assert function.global_value == 0
    # Validate function properties
    assert function.is_bounded() is False
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_bessel():
    function = func.Bessel()
    # Validate the default configuration
    assert function.get_dimensionality() == 1
    assert function.fast is False
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())
    # All same, but now for fast configuration
    function = func.Bessel(True)
    assert function.fast is True
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_modifiedbessel():
    function = func.ModifiedBessel()
    # Validate the default configuration
    assert function.get_dimensionality() == 1
    assert function.fast is False
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())
    # All same, but now for fast configuration
    function = func.ModifiedBessel(True)
    assert function.fast is True
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_eggbox():
    function = func.Eggbox()
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_multivariatenormal():
    function = func.MultivariateNormal()
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    assert np.array_equal(function.covariance, np.identity(2))
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)
    # Same, but now with a defined covariance matrix
    mat = np.random.rand(2, 2)
    mat = np.dot(mat, mat.transpose())
    function = func.MultivariateNormal(mat)
    assert np.array_equal(function.covariance, mat)
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_gaussianshells():
    function = func.GaussianShells()
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    assert np.array_equal(function.c_1, np.array([2.5, 0]))
    assert function.r_1 == 2.0
    assert function.w_1 == 0.1
    assert np.array_equal(function.c_2, np.array([-2.5, 0]))
    assert function.r_2 == 2.0
    assert function.w_2 == 0.1
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_linear():
    function = func.Linear()
    minima = [[0, 0]]
    value_minima = 0
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate minima
    y = function(np.array(minima))
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) == len(y)
    # Validate that these minima are indeed minima
    x = np.random.rand(10000, function.get_dimensionality())
    x *= function.ranges[:, 1] - function.ranges[:, 0]
    x += function.ranges[:, 0]
    y = function(x)
    assert np.sum(1.0*(np.around(y, 8) == value_minima)) <= len(y)
    # Validate output shape
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_reciprocal():
    function = func.Reciprocal()
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())*0.5+0.2
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_breitwigner():
    function = func.BreitWigner()
    # Validate the default configuration
    assert function.get_dimensionality() == 1
    assert function.m == 50
    assert function.width == 15
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is True
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    assert function(x, True).shape == (10000, function.get_dimensionality())


def test_testfunction_goldsteinprice():
    function = func.GoldsteinPrice()
    # Validate the default configuration
    assert function.get_dimensionality() == 2
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_schwefel():
    function = func.Schwefel()
    # Validate the default configuration
    assert function.get_dimensionality() == 5
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    # Validate output shape
    x = np.random.rand(10000, function.get_dimensionality())
    y = function(x)
    assert y.shape == (10000, 1)
    with pytest.raises(func.NoDerivativeError):
        function(x, True)


def test_testfunction_mssm7():
    function = func.MSSM7()
    # Validate the default configuration
    assert function.get_dimensionality() == 12
    # Validate function properties
    assert function.is_bounded() is True
    assert function.is_differentiable() is False
    with pytest.raises(func.NoDerivativeError):
        function(x, True)
