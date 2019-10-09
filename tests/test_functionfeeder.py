import numpy as np
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
    feeder.load_function("Sphere", {'dimensionality': 2})
    assert feeder.functions[0].dimensionality == 3
    assert feeder.functions[1].dimensionality == 5
    assert feeder.functions[2].dimensionality == 10
    assert feeder.functions[3].dimensionality == 2
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
