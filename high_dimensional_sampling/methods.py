from abc import ABCMeta, abstractmethod


class Sampler(Metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    
    @abstractmethod
    def is_finished(self):
        raise NotImplementedError