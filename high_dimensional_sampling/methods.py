from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def __init__(self):
        self.store_parameters = []

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    
    @abstractmethod
    def is_finished(self):
        raise NotImplementedError
