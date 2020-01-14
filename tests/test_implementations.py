import inspect
import shutil
from high_dimensional_sampling.posterior import RejectionSampling
from high_dimensional_sampling.optimisation import RandomOptimisation, Ampgo
from high_dimensional_sampling import functions as func
from high_dimensional_sampling import procedures as proc
from high_dimensional_sampling import experiments as exp


def test_posterior_implementations(tmp_path):
    classes = [RejectionSampling]
    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.PosteriorSamplingExperiment(procedure, str(tmp_path))
        experiment.run(func.Cosine(), finish_line=250)
        shutil.rmtree(str(tmp_path))


def test_optimisation_implementations(tmp_path):
    classes = [RandomOptimisation]
    for procedure_class in classes:
        procedure = procedure_class()
        experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
        experiment.run(func.Himmelblau(), finish_line=250)
        shutil.rmtree(str(tmp_path))
