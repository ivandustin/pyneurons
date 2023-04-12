from abc import ABC, abstractmethod
from .tuple import Tuple


class Model(Tuple, ABC):
    @abstractmethod
    def __call__(self):
        pass
