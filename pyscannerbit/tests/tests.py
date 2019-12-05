"""
Class representing HDF5 results from ScannerBit
===============================================
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


class HDF5(h5py.File):
    """
    """
    def __init__(self, file_name, model=None, loglike="LogLike", posterior="Posterior"):
        """
        """
        self.loglike = loglike
        self.posterior = posterior
        super(HDF5, self).__init__(file_name, mode='r')
        self.group = self.keys()[0]
        self.model = model if model else self.get_model_name()

    def get_model_name(self):
        """
        """
        for k in self[self.group].keys():
            if "::" in k:
                return k.split("::")[0]

    def get_param_names(self):
        """
        """
        prefix = "{}::".format(self.model)
        suffix = "_isvalid"
        return [k[len(prefix):] for k in self[self.group].keys()
            if k.startswith(prefix) and not k.endswith(suffix)]

    def get_param(self, name):
        """
        """
        try:
            valid = self[self.group]["{}_isvalid".format(name)]
        except KeyError:
            name = "{}::{}".format(self.model, name)
            valid = self[self.group]["{}_isvalid".format(name)]

        mask = np.array(valid, dtype=np.bool)
        return self[self.group][name][mask]

    def get_loglike(self):
        """
        """
        return self.get_param(self.loglike)

    def get_posterior(self):
        """
        """
        return self.get_param(self.posterior)

    def get_best_fit(self, name):
        """
        """
        loglike = self.get_loglike()
        index = np.argmax(loglike)
        return self.get_param(name)[index]

    def get_min_chi_squared(self):
        """
        """
        return -2. * self.get_loglike().max()

    def make_plot(self, name_x, name_y):
        """
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.scatter(self.get_param(name_x), self.get_param(name_y),
          marker="o", facecolor='k', edgecolor='', alpha=0.5, s=20)
        ax.scatter(self.get_best_fit(name_x), self.get_best_fit(name_y),
          marker="*", facecolor='Gold', alpha=1., s=250, label="Best-fit")

        ax.set_xlabel(name_x)
        ax.set_ylabel(name_y)
        ax.legend()
        plt.show()
