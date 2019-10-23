import pytest
import high_dimensional_sampling.functions as func


def test_functionfeeder_manual():
    # Create function feeder object
    feeder = func.FunctionFeeder()
    # Check if length property works
    assert len(feeder) == 0
    feeder.add_function(func.Linear())
    feeder.add_function(func.Linear())
    assert len(feeder) == 2
    # Check reset
    feeder.reset()
    assert len(feeder) == 0


def test_functionfeeder_loadbyname():
    # Create function feeder object
    feeder = func.FunctionFeeder()
    # Load function by name
    feeder.load_function("Linear")
    assert feeder.functions[0].name == "Linear"
    feeder.reset()
    # Load function with parameters
    feeder.load_function("Linear", {'name': 'TmpTestName'})
    assert feeder.functions[0].name == "TmpTestName"
    feeder.load_function("Sphere")
    feeder.load_function("Sphere", {'dimensionality': 5})
    feeder.load_function("Sphere", {'dimensionality': 10})
    feeder.load_function("Sphere", {'dimensionality': 3})
    assert feeder.functions[1].get_dimensionality() == 3
    assert feeder.functions[2].get_dimensionality() == 5
    assert feeder.functions[3].get_dimensionality() == 10
    assert feeder.functions[4].get_dimensionality() == 3
    feeder.reset()


def test_functionfeeder_names():
    # Create function feeder object
    feeder = func.FunctionFeeder()
    # Check if names of functions are not unique (as they should be)
    feeder.add_function(func.Linear())
    feeder.add_function(func.Linear())
    feeder.load_function("Linear")
    feeder.load_function("Linear")
    names = [f.name for f in feeder]
    uniques = []
    for n in names:
        if n not in uniques:
            uniques.append(n)
    assert len(uniques) == 1
    # Make names unique
    feeder.fix_duplicate_names()
    names = [f.name for f in feeder]
    uniques = []
    for n in names:
        if n not in uniques:
            uniques.append(n)
    assert len(uniques) == len(names)

def test_functionfeeder_groups():
    feeder = func.FunctionFeeder()
    # Check if can load all known groups
    for group in ['with_derivative', 'no_derivative', 'optimisation', 'posterior', 'bounded', 'unbounded', 'optimization']:
        feeder.load_function_group(group)
        feeder.reset()
    # Validate that bounded/unbounded and with_derivative/no_derivative indeed
    # consist only of functions that have those properties
    feeder.load_function_group('with_derivative')
    for function in feeder:
        assert function.is_differentiable() == True
    feeder.reset()
    feeder.load_function_group('no_derivative')
    for function in feeder:
        assert function.is_differentiable() == False
    feeder.reset()
    feeder.load_function_group('bounded')
    for function in feeder:
        assert function.is_bounded() == True
    feeder.reset()
    feeder.load_function_group('unbounded')
    for function in feeder:
        print(type(function).__name__, function.is_bounded())
        assert function.is_bounded() == False
    feeder.reset()
    # Check if exception is raised when unknown group is provided
    with pytest.raises(Exception):
        feeder.load_function_group('not_existing_group')
    with pytest.raises(Exception):
        feeder.load_function_group(123)
    # Check if can load multiple groups at the same time
    feeder.load_function_group(['optimisation', 'bounded'])
    feeder.reset()
    with pytest.raises(Exception):
        feeder.load_function_group(['optimisation', 'not_existing_group'])
    # Load function with parameters
    feeder.load_function_group('with_derivative', {
        'Sphere': {'dimensionality': 5}
    })
    for function in feeder:
        if type(function).__name__ == "Sphere":
            assert function.get_dimensionality() == 5

def test_load_function():
    feeder = func.FunctionFeeder()
    # Check error is raised when function is not known
    feeder.load_function('Sphere')
    with pytest.raises(Exception):
        feeder.load_function('non_existent_function')
    # Check error is raised when reference is not to a TestFunction
    with pytest.raises(Exception):
        feeder.load_function('FunctionFeeder')
    # Check error is raised when parameters are not fed as a dictionary
    feeder.load_function('Sphere', {'dimensionality': 3})
    with pytest.raises(Exception):
        feeder.load_function('Sphere', 3)
    with pytest.raises(Exception):
        feeder.load_function('Sphere', "dim")
    with pytest.raises(Exception):
        feeder.load_function('Sphere', {'dimension': 7})

def test_add_function():
    feeder = func.FunctionFeeder()
    # Let error be raised if non-TestFunction is added
    sphere = func.Sphere()
    feeder.add_function(sphere)
    feeder_2 = func.FunctionFeeder()
    with pytest.raises(Exception):
        feeder.add_function(feeder_2)