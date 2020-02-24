import numpy as np
import shutil
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import experiments as exp
from high_dimensional_sampling import optimisation as optim


class TmpExperimentCorrect(exp.Experiment):
    def make_metrics(self):
        return {}

    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


def test_hiddenfunctions_general():
    function = func.HiddenFunction1()
    assert function.packageloc is None
    assert function.funcname == 'test_func_1.bin'
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


def test_hiddenfunctions_use():
    procedure = optim.RandomOptimisation()
    path = "./tmpexperiment"
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.HiddenFunction1(), log_data=False)
    shutil.rmtree(path)
