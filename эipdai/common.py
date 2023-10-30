import numpy as np
from enum import IntEnum

N_PLAYERS = 3

N_OUTCOMES = 8

PAYOFF_MATRIX = np.array([
    [[6, 3], [3, 0]], 
    [[8, 5], [5, 2]]
])

class Actions(IntEnum):
    '''
    N - No action
    C - Cooperate
    D - Defect
    '''
    N = -1,
    C = 0,
    D = 1
