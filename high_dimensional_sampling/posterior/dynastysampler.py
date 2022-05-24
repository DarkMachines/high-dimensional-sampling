#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:40:38 2022

@author: dm
"""

import high_dimensional_sampling as hds

# basic numeric setup
import numpy as np
from numpy import linalg
from importlib import import_module
"""
Example of an optimisation experiment. Implemented procedure is explained at
https://en.wikipedia.org/wiki/Random_optimization
"""
"""
Dynesty usage:
    The sampler recieves 3 arguments: loglike,ptform,ndim
    where loglike is the log-likelihood function, ptform is the prior
    transform, and ndim the dimensionality of the problem.
    
    Dynesty sampler is documented here:
    https://dynesty.readthedocs.io/en/latest/api.html?highlight=dynesty.NestedSampler#dynesty.dynesty.NestedSampler
    

"""


class DynestySampler(hds.Procedure):
    def __init__(self,
                 n_initial=10,
                 n_sample=10,
                 converge: bool = False,
                 **kwargs):
        self.store_parameters = ['n_initial', 'n_sample']
        self.n_initial = n_initial
        self.n_sample = n_sample
        self.dynasty = import_module("dynasty")
        self.sampler = self.dynesty.NestedSampler(
            loglikelihood=self.__DefaultLogLike__,
            prior_transform=self.__Default_prior_transform,
            ndim=3)  #TBD
        self.reset()
        self._rstate = np.random.default_rng(5647)
        self._res = None
        self._function = 0
        self._converge = converge
        self._kwargs = {
            'bound': None,
            'method': None,
            'update_interval_ratio': None,
            'first_update': None,
            'rstate': None,
            'queue_size': None,
            'pool': None,
            'use_pool': None,
            'ncdim': None,
            'nlive0': None,
            'kwargs': None
        }
        self._comstring = ''
        for key in kwargs:
            try:
                self._kwargs[key] = kwargs[key]
                self._comstring += ',' + str(key) + '=' + str(kwargs[key])
            except Exception as e:
                print(
                    "Exception: most probably you didn't use a correct keyword in **kwargs"
                )
                raise e

    def __DefaultLogLike__(self, x):
        ndim = 3  # number of dimensions
        C = np.identity(ndim)  # set covariance to identity matrix
        C[C == 0] = 0.95  # set off-diagonal terms (strongly correlated)
        Cinv = linalg.inv(C)  # precision matrix
        lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(linalg.det(C))
                        )  # ln(normalization)

        # 3-D correlated multivariate normal log-likelihood

        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

    def __Default_prior_transform(self, u):
        """Transforms our unit cube samples `u` to a flat prior between -10. and 10. in each variable."""
        return 1. * (2. * u - 1.)

    def prior_transform(self, u):
        return (u * self.ranges + self.mins)

    def _CreateSampler(self, function):
        minmax = np.array(function.get_ranges(0.01))
        self.ranges = minmax[:, 1] - minmax[:, 0]
        self.mins = minmax[:, 0]
        command = """self.sampler = self.dynesty.NestedSampler(loglikelihood=function,prior_transform=self.prior_transform,ndim=function.get_dimensionality(),nlive=self.n_sample"""
        command += self._comstring + ')'
        print(command)
        exec(command)
        self._function = function
        print(self.sampler.bound)

    def __call__(self, function):
        """
            Here we first create import the test function into the sampler,
        """
        if self._function != function:
            self._CreateSampler(function)
        try:
            self.sampler.run_nested(maxiter=1, maxcall=1, print_progress=False)
            self._res = self.sampler.results

        except Exception as e:
            print('exeption occured')
            pass
        #self._FirstCall=True

        # Get ranges of the test function. The 0.001 moves the minima 0.001 up
        # and the maxima 0.001 down, in order to make use the sampling is not
        # by accident moving outside of the test function range.

        return (self._res['samples'], self._res['logz'])

    def get_initial_position(self, ranges, n_sample_initial):
        ndim = len(ranges)
        r = np.array(ranges)
        x = np.random.rand(n_sample_initial, ndim)
        x = x * (r[:, 1] - r[:, 0]) + r[:, 0]
        return x

    def get_point(self, ranges, stdev=0.01, n_sample=1):
        cov = np.identity(len(ranges)) * stdev
        return np.random.multivariate_normal(self.current_position[0], cov,
                                             n_sample)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):

        return (self._converge
                and self.dynesty.dynamicsampler.stopping_function(
                    self.sampler.results))
        #return False

    def check_testfunction(self, function):
        return True


"""
 Usage example

"""
kwargs = {'bound': '"balls"'}

procedure = DynestySampler(n_initial=1000,
                           n_sample=1000,
                           converge=True,
                           **kwargs)
experiment = hds.OptimisationExperiment(procedure, './hds')
feeder = hds.functions.FunctionFeeder()
feeder.load_function_group(['optimisation', 'bounded'])

for function in feeder:
    experiment.run(function, finish_line=10000000)
