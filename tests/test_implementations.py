import inspect
import shutil
from high_dimensional_sampling import posterior
from high_dimensional_sampling import optimisation
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import procedures as proc
from high_dimensional_sampling import experiments as exp


def test_posterior_implementations(tmp_path):
    all_classes = inspect.getmembers(posterior, inspect.isclass)
    classes = [c[1] for c in all_classes if isinstance(c[1](), proc.Procedure)]
    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.PosteriorSamplingExperiment(procedure, str(tmp_path))
        experiment.run(func.Cosine(), finish_line=250)
        shutil.rmtree(str(tmp_path))


def test_optimisation_implementations(tmp_path):
    all_classes = inspect.getmembers(optimisation, inspect.isclass)
    # Exempt Pyscannerbit interface due to issues with PS install on Travis
    filtered = [c for c in all_classes if c[0] is not 'PyScannerBit']
    classes = [c[1] for c in filtered if isinstance(c[1](), proc.Procedure)]
    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
        experiment.run(func.Himmelblau(), finish_line=250)
        shutil.rmtree(str(tmp_path))
