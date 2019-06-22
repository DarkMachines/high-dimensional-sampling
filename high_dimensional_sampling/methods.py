from abc import ABC, abstractmethod


class Method(ABC):
    @abstractmethod
    def __init__(self):
        self.store_parameters = []

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    
    @abstractmethod
    def is_finished(self):
        raise NotImplementedError
