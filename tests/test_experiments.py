import os
import shutil
import yaml
import time
import pytest
import numpy as np
import pandas as pd
from high_dimensional_sampling import experiments as exp
from high_dimensional_sampling import procedures as proc
from high_dimensional_sampling import functions as func


class TmpProcedure(proc.Procedure):
    def __init__(self):
        self.store_parameters = ['a']
        self.a = 10
        self.allow_function = False

    def __call__(self, function):
        return (np.random.rand(100, 3), np.random.rand(100, 1))

    def is_finished(self):
        return False

    def check_testfunction(self, function):
        return self.allow_function

    def reset(self):
        pass


class TmpExperimentCorrect(exp.Experiment):
    def make_metrics(self):
        return {}

    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


class TmpExperimentWrong1(exp.Experiment):
    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


class TmpExperimentWrong2(exp.Experiment):
    def make_metrics(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


class TmpExperimentWrong3(exp.Experiment):
    def make_metrics(self):
        pass

    def _event_start_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


class TmpExperimentWrong4(exp.Experiment):
    def make_metrics(self):
        pass

    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass


def test_experiment_initialisation():
    procedure = TmpProcedure()
    path = "./tmpexperiment"
    experiment = TmpExperimentCorrect(procedure, path)
    # Test if properties are correctly set
    assert experiment.path == path
    assert experiment.procedure == procedure
    assert experiment.logger is None
    # Test if Experiment class is correctly abstracted
    with pytest.raises(TypeError):
        _ = exp.Experiment(procedure, path)  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpExperimentWrong1(procedure, path)  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpExperimentWrong2(procedure, path)  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpExperimentWrong3(procedure, path)  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpExperimentWrong4(procedure, path)  # pylint: disable=E0110
    # Check if procedure is instance of Procedure check works
    with pytest.raises(Exception):
        _ = TmpExperimentCorrect("tmp", "./tmpexperiment")


def test_experiment_stopcriterion():
    procedure = TmpProcedure()
    path = "./tmpexperiment"
    experiment = TmpExperimentCorrect(procedure, path)
    # Test if stopping criterion is correctly triggered
    experiment.n_sampled = 0
    experiment.finish_line = 1001
    x, y = np.random.rand(1002, 2), np.random.rand(1000, 1)
    assert experiment._stop_experiment(x, y) is True
    experiment.n_sampled = 0
    assert experiment._stop_experiment(x[:-1], y[:-1]) is True
    experiment.n_sampled = 0
    assert experiment._stop_experiment(x[:-2], y[:-2]) is False


def test_experiment_perform():
    procedure = TmpProcedure()
    path = "./tmpexperiment"
    experiment = TmpExperimentCorrect(procedure, path)
    # Test if provided function for performing experiment needs to be a
    # TestFunction (which it should)
    with pytest.raises(Exception):
        experiment._perform_experiment("not-a-testfunction")
    # Check if function can be tested for validity by child class through the
    # .check_testfunction method
    with pytest.raises(Exception):
        experiment._perform_experiment(func.GaussianShells())
    # Perform experiment and check if all folders and files are created
    procedure.allow_function = True
    experiment.run(func.GaussianShells())
    assert experiment.finish_line == 1000
    assert os.path.exists(path + os.sep + 'gaussianshells') is True
    assert os.path.exists(path + os.sep + 'gaussianshells/samples.csv') is True
    assert os.path.exists(path + os.sep +
                          'gaussianshells/functioncalls.csv') is True
    assert os.path.exists(path + os.sep +
                          'gaussianshells/procedurecalls.csv') is True
    assert os.path.exists(path + os.sep + 'benchmarks.yaml') is True
    # Repeat experiment, but now disable logging of datapoints
    shutil.rmtree(path)
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.GaussianShells(), finish_line=90, log_data=False)
    assert experiment.finish_line == 90
    assert os.path.getsize(path + os.sep + 'gaussianshells/samples.csv') < 100
    shutil.rmtree(path)


def test_experiment_optimisation():
    procedure = TmpProcedure()
    path = "./tmpexperiment"
    experiment = exp.OptimisationExperiment(procedure, path)
    # Test start_experiment event
    experiment._event_start_experiment()
    assert experiment.best_x is None
    assert experiment.best_y is None
    # Test that end experiment event does return nothing
    # TODO: how to test a function that does nothing?
    assert experiment._event_end_experiment() is None
    x = np.random.rand(100, 2)
    y = x[:, 1]
    m = np.argmin(y)
    experiment._event_new_samples(x, y)
    assert np.array_equal(experiment.best_x[0], x[m]) is True
    assert np.array_equal(experiment.best_y[0], y[m]) is True
    # Test make metrics function
    metrics = experiment.make_metrics()
    assert isinstance(experiment.make_metrics(), dict)
    assert metrics['best_point'] == x[m].tolist()
    assert metrics['best_value'] == y[m].tolist()


def test_experiment_posteriorsampling():
    procedure = TmpProcedure()
    path = "./tmpexperiment"
    experiment = exp.PosteriorSamplingExperiment(procedure, path)
    # Test make metrics function
    assert isinstance(experiment.make_metrics(), dict)
    assert len(experiment.make_metrics()) == 0
    # Test if all event methods dont do anything
    # TODO: how to test a function that does nothing?
    assert experiment._event_start_experiment() is None
    assert experiment._event_end_experiment() is None
    x, y = np.random.rand(100, 2), np.random.rand(100, 1)
    assert experiment._event_new_samples(x, y) is None


def test_experiment_logyaml():
    procedure = TmpProcedure()
    procedure.allow_function = True
    path = "./tmpexperiment"
    experiment = TmpExperimentCorrect(procedure, path)
    # Test if the logbook contains all entries
    function = func.GaussianShells()
    experiment.run(function)
    log = {}
    with open(path + "/gaussianshells/experiment.yaml", 'r') as stream:
        log = yaml.load(stream)
    assert 'experiment' in log
    assert 'type' in log['experiment']
    assert 'function' in log
    assert 'name' in log['function']
    assert 'properties' in log['function']
    assert 'ranges' in log['function']['properties']
    assert log['function']['properties']['ranges'] == function.ranges
    assert 'testfunction' in log['function']
    assert 'meta' in log
    assert 'datetime' in log['meta']
    assert 'timestamp' in log['meta']
    assert 'user' in log['meta']
    assert 'procedure' in log
    assert 'name' in log['procedure']
    assert 'properties' in log['procedure']
    assert log['procedure']['properties'] == {'a': 10}
    shutil.rmtree(path)


def test_logger_initialisation_deletion():
    basepath = '.'
    subfolder = 'tmplog'
    log = exp.Logger(basepath, subfolder)
    # Check if intialisation works properly
    assert log.basepath == '.'
    assert log.path == './tmplog'
    assert log.procedure_calls == 0
    assert log.create_samples_header is True
    # Check that folder exists
    assert os.path.exists(basepath + os.sep + subfolder) is True
    # Check that handles for 3 files are opened
    assert os.path.exists(basepath + os.sep + subfolder + os.sep +
                          'samples.csv') is True
    assert os.path.exists(basepath + os.sep + subfolder + os.sep +
                          'functioncalls.csv') is True
    assert os.path.exists(basepath + os.sep + subfolder + os.sep +
                          'procedurecalls.csv') is True
    # Check that duplicate names are handled properly
    log2 = exp.Logger(basepath, subfolder)
    assert log.path != log2.path
    # Remove created folders
    shutil.rmtree(log.path)
    shutil.rmtree(log2.path)
    # Close handles
    log.handle_samples.close()
    del (log.handle_samples)
    del (log)


def test_logger_logsamples():
    basepath = '.'
    subfolder = 'tmplog'
    log = exp.Logger(basepath, subfolder)
    # Log samples
    x = np.random.rand(1000, 3)
    y = np.random.rand(1000, 1).reshape(-1, 1)
    total = np.hstack((x, y))
    log.log_samples(x, y)
    # Read log
    data = np.genfromtxt(basepath + os.sep + subfolder + os.sep +
                         'samples.csv',
                         skip_header=1,
                         delimiter=',')
    assert np.array_equal(total, data[:, 1:]) is True
    # Add new data
    log.log_samples(x, y)
    # Read log
    data = np.genfromtxt(basepath + os.sep + subfolder + os.sep +
                         'samples.csv',
                         delimiter=',',
                         skip_header=1).astype(np.float)
    assert np.array_equal(np.vstack((total, total)), data[:, 1:]) is True
    shutil.rmtree(basepath + os.sep + subfolder)


def test_logger_procedurecalls():
    basepath = '.'
    subfolder = 'tmplog'
    log = exp.Logger(basepath, subfolder)
    # Fake procedure calls
    log.procedure_calls = 10
    # Log procedure calls
    log.log_procedure_calls(5, 9, 3)
    log.log_procedure_calls(1, 2, 6)
    # Read procedure call log
    call_log = np.genfromtxt(log.path + os.sep + 'procedurecalls.csv',
                             delimiter=',',
                             skip_header=1).astype(np.float)
    reference = np.array([[10, 5, 9, 3], [10, 1, 2, 6]])
    assert np.array_equal(call_log, reference)
    # Read procedure calls and assert that they are what i think they should be
    shutil.rmtree(basepath + os.sep + subfolder)


def test_logger_functioncalls():
    basepath = '.'
    subfolder = 'tmplog'
    log = exp.Logger(basepath, subfolder)
    # Fake function calls
    log.procedure_calls = 10
    function = func.GaussianShells()
    function.counter = [[10, 3, 1], [9, 2, 0]]
    log.log_function_calls(function)
    # Read function call log
    call_log = pd.read_csv(log.path + os.sep + 'functioncalls.csv')
    reference = np.array([[10, 10, 3, 1], [10, 9, 2, 0]])
    assert np.array_equal(call_log.values[:, :-1], reference[:, :-1])
    assert np.array_equal(call_log['asked_for_derivative'] * 1,
                          reference[:, -1])
    shutil.rmtree(log.path)


def test_logger_benchmarks():
    basepath = '.'
    subfolder = 'tmplog'
    log = exp.Logger(basepath, subfolder)
    # Create benchmarks
    log.log_benchmarks()
    # Read benchmarks and validate they are > 0
    benchmarks = {}
    with open(basepath + "/benchmarks.yaml", 'r') as stream:
        benchmarks = yaml.load(stream)
    assert 'benchmarks' in benchmarks
    assert 'matrix_inversion' in benchmarks['benchmarks']
    assert 'sha_hashing' in benchmarks['benchmarks']
    assert benchmarks['benchmarks']['matrix_inversion'] > 0
    assert benchmarks['benchmarks']['sha_hashing'] > 0
    # Test that benchmark file is not made a second time
    t_start = time.time()
    log.log_benchmarks()
    assert time.time() - t_start < 1
    # Remove logs
    os.remove(basepath + os.sep + 'benchmarks.yaml')
    shutil.rmtree(log.path)
