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


def test_optimisation_randomoptimisation(tmp_path):
    procedure = optimisation.RandomOptimisation(),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))


def test_optimisation_particlefilter(tmp_path):
    procedure = optimisation.ParticleFilter(),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))


def test_optimisation_gpyopt(tmp_path):
    procedure = optimisation.GPyOpt(),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))


def test_optimisation_cmae(tmp_path):
    procedure = optimisation.CMAOptimisation(),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))


def test_optimisation_ampgo(tmp_path):
    procedure = optimisation.Ampgo(),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))


def test_optimisation_turbo(tmp_path):
    procedure = optimisation.TuRBO(max_evals=5, n_training_steps=5),
    experiment = exp.OptimisationExperiment(procedure, str(tmp_path))
    experiment.run(func.Himmelblau(), finish_line=250)
    shutil.rmtree(str(tmp_path))
