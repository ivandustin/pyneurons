from abc import ABC, abstractmethod
from pyneurons.classes import Tuple


class Model(Tuple, ABC):
    @abstractmethod
    def __call__(self):
        pass
