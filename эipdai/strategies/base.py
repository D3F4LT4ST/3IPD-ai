import numpy as np
from abc import ABC, abstractmethod
from ..common import Actions

class Strategy(ABC):
    '''
    Base class for all strategies.
    '''
    def __init__(self, name: str=None):
        self._name = name if name else self.__class__.__name__

    def __str__(self) -> str:
        return self._name

    @abstractmethod
    def play(
        self, 
        observation: np.array,
        reward: int=None,
    ) -> Actions:
        '''
        Returns action for the current observation.

        Args:
            observation: current observation
            reward: most recent reward

        Returns: action
        '''

    def reset(self):
        pass