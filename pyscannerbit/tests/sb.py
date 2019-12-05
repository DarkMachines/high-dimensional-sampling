"""
Pythonic interface to ScannerBit
================================
"""

import sys
import ctypes
import inspect

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import pyscannerbit
import default
import import_yaml
from hdf5_reader import HDF5


DIR = "runs/test_scan/samples/"


class Scan(object):
    """
    """
    def __init__(self, function, bounds, prior_types=None, kwargs=None, scanner="multinest",
      settings=None, yaml="scan.yaml"):
        """
        """
        self.function = function
        self.bounds = bounds
        self.prior_types = prior_types if prior_types else ["flat"] * len(bounds)
        self.scanner = scanner
        self.settings = settings if settings else default.settings
        self.yaml = yaml
        self.kwargs = kwargs

        signature = inspect.getargspec(self.function)
        n_kwargs = len(signature.defaults or [])
        self._argument_names = signature.args[:-n_kwargs or None]
        assert len(self._argument_names) == len(self.bounds)

        self._model_name = self.settings["Parameters"].keys()[0]
        self._wrapped_function = self._wrap_function()
        self._scanned = False
        self._write_yaml()

    def _get_hdf5_name(self):
        """
        """
        assert self.settings["Printer"]["printer"] == "hdf5"
        file_name = self.settings["Printer"]["options"]["output_file"]
        return "{}/{}".format(DIR, file_name)

    def _write_yaml(self):
        """
        """
        self.settings["Scanner"]["use_scanner"] = self.scanner
        self.settings["Parameters"][self._model_name] = dict()
        for n, b, t in zip(self._argument_names, self.bounds, self.prior_types):
            self.settings["Parameters"][self._model_name][n] = {"range": b, "prior_type": t}

        with open(self.yaml, 'w') as f:
            import_yaml.dump(self.settings, stream=f)

    def _wrap_function(self):
        """
        """
        def wrapped_function(par_dict):
            """
            """
            arguments = [par_dict["{}::{}".format(self._model_name, n)]
              for n in self._argument_names]
            return self.function(*arguments, **(self.kwargs or {}))

        return wrapped_function

    def scan(self):
        """
        """
        pyscannerbit.run_scan(self.yaml, self._wrapped_function)
        self._scanned = True

    def get_hdf5(self):
        """
        """
        assert self._scanned
        return HDF5(self._get_hdf5_name(), model=self._model_name)
