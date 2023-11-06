import random
import numpy as np
from .base import Strategy
from ..common import Actions, N_OUTCOMES
from typing import List

class GeneticStrategy(Strategy):

    def __init__(
        self, 
        memory_len: int,
        genotype: List[int]=None,
        name: str=None,
    ):
        super().__init__(name)

        self._memory_len = memory_len

        self._outcomes_coef = np.array([N_OUTCOMES**i for i in range(memory_len-1,-1,-1)])
        self._moves_coef = np.array([2**i for i in range(0,memory_len)])

        if genotype is None:
            genotype_len = sum([N_OUTCOMES ** history_len for history_len in range(memory_len, -1, -1)])
            self._genotype = [random.randint(Actions.C, Actions.D) for _ in range(genotype_len)]
        else:
            self._genotype = genotype

    @property
    def genotype(self):
        return self._genotype
    
    def play(
        self, 
        observation: np.array, 
        reward: int = None
    ) -> Actions:
        
        obs_trimmed = observation[:, -self._memory_len:]
        obs_masked = np.ma.masked_array(obs_trimmed, obs_trimmed<0)

        no_history_offset = ((obs_masked>=0).any(axis=0) * self._outcomes_coef).sum()

        gene_idx = ((obs_masked * self._moves_coef[:, None]).sum(axis=0) * self._outcomes_coef).sum() + no_history_offset
        if type(gene_idx) is np.ma.core.MaskedConstant:
            gene_idx = 0

        return self._genotype[gene_idx]