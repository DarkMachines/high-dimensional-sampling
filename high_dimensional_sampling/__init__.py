# -*- coding: utf-8 -*-

from .__version__ import __version__  # noqa: F401
from .experiments import (PosteriorSamplingExperiment,  # noqa: F401
                          OptimisationExperiment,
                          OptimizationExperiment)
from .procedures import Procedure  # noqa: F401
from . import functions  # noqa: F401

__author__ = "Joeri Hermans, Bob Stienen"
__email__ = 'joeri.hermans@doct.uliege.be, b.stienen@science.ru.nl'
