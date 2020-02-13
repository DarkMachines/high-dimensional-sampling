import os
import numpy as np
import high_dimensional_sampling as hds
import pypolychord as pc
from anesthetic import NestedSamples
from pypolychord.settings import PolyChordSettings


class PolyChord(hds.Procedure):
    def __init__(self):
        self.store_parameters = []
        self._is_finished = False
        self._reset = False

    def __call__(self, function):
        ranges = function.get_ranges()

        def prior(cube):
            return cube * ranges[:, 1] + (1-cube) * ranges[:, 0]

        def likelihood(theta):
            return function([theta])[0, 0], []

        def dumper(live, dead, logweights, logZ, logZerr):
            pass

        nDims = function.get_dimensionality()
        nDerived = 0
        settings = PolyChordSettings(nDims, nDerived)
        settings.base_dir = '.polychord_chains'
        settings.file_root = function.name
        settings.posteriors = False
        settings.equals = False
        settings.write_live = False
        settings.write_prior = False
        settings.write_paramnames = False
        settings.write_stats = True
        settings.write_resume = True
        settings.read_resume = True

        if self._reset:
            self._reset = False
            settings.read_resume = False

        if not self.is_finished():
            pc.run_polychord(likelihood, nDims, nDerived,
                             settings, prior, dumper)

        self._is_finished = True
        root = os.path.join(settings.base_dir, settings.file_root)
        samples = NestedSamples(root=root)
        x = samples.iloc[:, :nDims].to_numpy()
        y = np.array([samples.logL.to_numpy()]).T

        return (x, y)

    def reset(self):
        self._is_finished = False
        self._reset = True

    def is_finished(self):
        return self._is_finished

    def check_testfunction(self, function):
        return True
