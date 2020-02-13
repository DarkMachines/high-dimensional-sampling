import inspect
import shutil
from high_dimensional_sampling import posterior
from high_dimensional_sampling import optimisation
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import experiments as exp
from high_dimensional_sampling import procedures as proc


def test_posterior_implementations(tmp_path):
    all_classes = inspect.getmembers(posterior, inspect.isclass)
    classes = [c[1] for c in all_classes if isinstance(c[1](), proc.Procedure)]
    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.PosteriorSamplingExperiment(procedure, str(tmp_path))
        experiment.run(func.Cosine(), finish_line=250)
        shutil.rmtree(str(tmp_path))


def test_optimisation_implementations(tmp_path):
    # These are the classes to test
    classes = [
        optimisation.RandomOptimisation,
        optimisation.ParticleFilter,
        optimisation.GPyOpt,
        optimisation.CMAOptimisation,
        optimisation.Ampgo,
        # optimisation.Pygmo,  # excluded because of a mandatory install
        # optimisation.PyScannerBit  # excluded because of a mandatory install
    ]

    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
        experiment.run(func.Himmelblau(), finish_line=250)
        shutil.rmtree(str(tmp_path))
