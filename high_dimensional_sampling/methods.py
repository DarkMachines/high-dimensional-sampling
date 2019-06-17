from abc import ABCMeta, abstractmethod


class Sampler(Metaclass=ABCMeta):
    def __call__(self):
        raise NotImplementedError
    
    def is_finished(self):
        raise NotImplementedError
    
    def initialise(self):
        raise NotImplementedError