import os
import shutil
import pytest
import numpy as np
import pandas as pd
from high_dimensional_sampling import results
from high_dimensional_sampling import experiments as exp
from high_dimensional_sampling import procedures as proc
from high_dimensional_sampling import functions as func


class TmpProcedure(proc.Procedure):
    def __init__(self):
        self.store_parameters = ['a']
        self.a = 10

    def __call__(self, function):
        return (np.random.rand(100, 3), np.random.rand(100, 1))

    def is_finished(self):
        return False

    def check_testfunction(self, function):
        return True

    def reset(self):
        pass


class TmpExperimentCorrect(exp.Experiment):
    def make_metrics(self):
        return {'x': 9, 'best_value': [3, 1]}

    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


class TmpExperimentCorrect2(exp.Experiment):
    def make_metrics(self):
        return {'x': 9}

    def _event_start_experiment(self):
        pass

    def _event_end_experiment(self):
        pass

    def _event_new_samples(self, x, y):
        pass


def test_results_result():
    path = "./tmpresultsexp1"
    # Test that if path does not exist, exception is raised
    with pytest.raises(Exception):
        _ = results.Result(path)
    # Create experiment
    procedure = TmpProcedure()
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.GaussianShells())
    # Test that results can now be properly be initialised, regardless of
    # directory seperator at end of path
    r = results.Result(path)
    _ = results.Result(path + '/')
    assert r.path == path + os.sep
    # Test that if benchmarks file is missing, results cannot be read
    os.remove(path + '/benchmarks.yaml')
    with pytest.raises(Exception):
        _ = results.Result(path)
    shutil.rmtree(path)


def test_results_submethods():
    # Create experiment
    procedure = TmpProcedure()
    path = "./tmpresultsexp2"
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.GaussianShells(), finish_line=1000)
    r = results.Result(path)
    # Check get_function_information
    assert r.get_function_information('testname') == ('testname', 0)
    assert r.get_function_information('testname_1') == ('testname', 1)
    assert r.get_function_information('test_name_3') == ('test_name', 3)
    # Check read_procedure_calls
    procedure_calls = r.read_procedure_calls('gaussianshells')
    assert isinstance(procedure_calls, pd.DataFrame) is True
    assert len(procedure_calls) == 10
    # Check procedure_information
    time = r.extract_procedure_information(procedure_calls)
    assert time == procedure_calls['dt'].sum()
    # Check read_experiment_yaml
    exp = r.read_experiment_yaml('gaussianshells')
    assert isinstance(exp, dict)
    assert len(exp) > 0
    assert 'meta' in exp
    # Check extract_result_information
    res = r.extract_result_information(exp)
    assert isinstance(res, dict)
    assert 'experiment_type' in res
    assert 'x' in res
    assert res['procedure_name'] == 'TmpProcedure'
    assert res['best_value'] == 3
    # Remove folder
    shutil.rmtree(path)


def test_results_df():
    # Create experiment
    procedure = TmpProcedure()
    path = "./tmpresultsexp3"
    experiment = TmpExperimentCorrect2(procedure, path)
    experiment.run(func.GaussianShells(), finish_line=1000)
    experiment.run(func.Sphere(), finish_line=1000)
    r = results.Result(path)
    # Get df
    df = r.get_results()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # Create df through function
    df2 = results.make_dataframe({'standard': path})
    print(df2)
    del (df2['experiment'])
    assert df.equals(df2)
    # Check if indicating multiple folders works
    _ = results.make_dataframe({'standard': path, 'other': path})
    # Remove folder
    shutil.rmtree(path)


def test_results_tables():
    # Create experiment
    procedure = TmpProcedure()
    path = "./tmpresultsexp4"
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.GaussianShells(), finish_line=1000)
    experiment.run(func.Sphere(), finish_line=1000)
    experiment.run(func.GaussianShells(), finish_line=1000)
    experiment.run(func.Sphere(), finish_line=1000)
    # Get results df
    df = results.make_dataframe({'standard': path, 'other': path})

    # Check that exceptions are raised correctly
    with pytest.raises(Exception):
        _, _ = results.tabulate_result(df, 'x', 'not-an-experiment')
    with pytest.raises(Exception):
        _, _ = results.tabulate_result(df, 'y', 'standard')
    # Check that the method works in general
    content, rows = results.tabulate_result(df, 'x', 'other')
    # Check content and rows format
    assert rows.tolist() == ['gaussianshells', 'sphere']
    assert isinstance(content, list)
    # Check that the method works with functions specified
    _, _ = results.tabulate_result(df, 'x', 'other', ['Sphere'])
    # Check that results can be stored to csv and tex (validate csv)
    _, _ = results.tabulate_result(df, 'x', 'other', path='./tmp_out.csv')
    _, _ = results.tabulate_result(df, 'x', 'other', path='./tmp_out.tex')
    _ = pd.read_csv('./tmp_out.csv')
    # Validate that allowed table output formats are only csv and tex
    _ = results.create_table_string(content, rows, ['x'], 'csv', '')
    _ = results.create_table_string(content, rows, ['x'], 'tex', '')
    with pytest.raises(Exception):
        _ = results.create_table_string(content, rows, ['None'], 'txt', '')
    # Remove folder and output files
    os.remove('./tmp_out.csv')
    os.remove('./tmp_out.tex')

    # Check that exceptions are raised correctly
    print(df)
    with pytest.raises(Exception):
        _, _, _ = results.tabulate_all_aggregated(
            df, 'time', 'mean', experiment_names=['not-an-experiment'])
    with pytest.raises(Exception):
        _, _, _ = results.tabulate_all_aggregated(df, 'y', 'mean')
    _, _, _ = results.tabulate_all_aggregated(df,
                                              'time',
                                              'mean',
                                              functions=['sphere'])
    _, _, _ = results.tabulate_all_aggregated(df,
                                              'time',
                                              'mean',
                                              experiment_names=['standard'])
    _, _, _ = results.tabulate_all_aggregated(df,
                                              'time',
                                              'mean',
                                              path='./tmp_out.csv')
    _ = pd.read_csv('./tmp_out.csv')
    content, rows, cols = results.tabulate_all_aggregated(df,
                                                          'time',
                                                          'mean',
                                                          path='./tmp_out.tex')
    assert isinstance(rows, list) is True
    assert isinstance(cols, list) is True
    assert isinstance(content, list)

    # Remove folder and output files
    os.remove('./tmp_out.csv')
    os.remove('./tmp_out.tex')
    shutil.rmtree(path)


def test_results_plots():
    # Create experiment
    procedure = TmpProcedure()
    path = "./tmpresultsexp5"
    experiment = TmpExperimentCorrect(procedure, path)
    experiment.run(func.GaussianShells(), finish_line=1000)
    experiment.run(func.Sphere(), finish_line=1000)
    # Get results df
    df = results.make_dataframe({'standard': path, 'other': path})

    # Boxplot experiment
    with pytest.raises(Exception):
        results.boxplot_experiment(df, 'time', 'not-an-experiment')
    with pytest.raises(Exception):
        results.boxplot_experiment(df, 'not-a-metric', 'standard')
    results.boxplot_experiment(df,
                               'time',
                               'standard',
                               logarithmic=True,
                               path='./plot.png')
    assert os.path.exists('./plot.png') is True
    results.boxplot_experiment(df,
                               'time',
                               'standard',
                               logarithmic=False,
                               path='./plot.png')
    os.remove('./plot.png')

    # Boxplot function
    with pytest.raises(Exception):
        results.boxplot_function(df, 'time', 'not-a-function')
    with pytest.raises(Exception):
        results.boxplot_function(df, 'not-a-metric', 'sphere')
    results.boxplot_function(df,
                             'time',
                             'sphere',
                             logarithmic=True,
                             path='./plot.png')
    assert os.path.exists('./plot.png') is True
    results.boxplot_function(df,
                             'time',
                             'sphere',
                             logarithmic=False,
                             path='./plot.png')
    os.remove('./plot.png')

    # Histogram experiment
    with pytest.raises(Exception):
        results.histogram_experiment(df, 'time', 'not-an-experiment')
    with pytest.raises(Exception):
        results.histogram_experiment(df, 'not-a-metric', 'standard')
    results.histogram_experiment(df,
                                 'time',
                                 'standard',
                                 logarithmic=True,
                                 path='./plot.png')
    assert os.path.exists('./plot.png') is True
    results.histogram_experiment(df,
                                 'time',
                                 'standard',
                                 logarithmic=False,
                                 path='./plot.png')
    os.remove('./plot.png')

    # Histogram function
    with pytest.raises(Exception):
        results.histogram_function(df, 'time', 'not-a-function')
    with pytest.raises(Exception):
        results.histogram_function(df, 'not-a-metric', 'sphere')
    results.histogram_function(df,
                               'time',
                               'sphere',
                               logarithmic=True,
                               path='./plot.png')
    assert os.path.exists('./plot.png') is True
    results.histogram_function(df,
                               'time',
                               'sphere',
                               logarithmic=False,
                               path='./plot.png')
    os.remove('./plot.png')

    shutil.rmtree(path)
