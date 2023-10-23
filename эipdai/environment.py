import random
import functools
import numpy as np
import gymnasium
import pettingzoo
from typing import Any, Union
from .common import Actions, N_PLAYERS, PAYOFF_MATRIX

class IPDGame(pettingzoo.ParallelEnv):
    '''
    Three player iterated prisoner's dilemma environment
    '''
    metadata = {'render_modes' : ['human'], 'name' : 'IPDGame'}

    def __init__(
        self,
        min_rounds: int,
        max_rounds: int,
        render_mode=None
    ) -> None:
        
        self._min_rounds = min_rounds
        self._max_rounds = max_rounds
        self.render_mode = render_mode

        self.possible_agents = [f'player_{i}' for i in range(N_PLAYERS)]

        self._action_space = gymnasium.spaces.Discrete(len([Actions.C, Actions.D]))

        self._observation_space = gymnasium.spaces.Box(
            low=Actions.N.value, high=Actions.D.value, shape=(N_PLAYERS, self._max_rounds), dtype=np.byte
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._action_space
    
    def step(self, actions: dict) -> tuple[dict, dict[str, int], dict[str, bool], dict[str, bool], dict[str, dict]]:
        self._round += 1

        actions = np.array(list(actions.values()))

        self._history = np.concatenate([self._history[:, 1:], actions[:,None]], axis=1)
        
        observations = {
            self.agents[i]: np.concatenate([self._history[i:], self._history[:i]])
            for i in range(len(self.agents))
        }

        rewards = {
            self.agents[i]: PAYOFF_MATRIX[tuple(np.concatenate([actions[i:], actions[:i]]))]
            for i in range(len(self.agents))
        } 

        terminations = {agent: self._round == self._end_round for agent in self.agents}
        trunctations = {agent: False for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, trunctations, infos

    def reset(self, seed: int=None, options: dict=None) -> tuple[dict, dict[Any, dict]]:
        if seed: random.seed(seed)

        self.agents = self.possible_agents[:]

        self._round = 0 
        self._end_round = random.randint(self._min_rounds, self._max_rounds)
        self._history = np.full([N_PLAYERS, self._max_rounds], Actions.N.value)

        observations = {agent: self._history for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def render(self) -> Union[np.ndarray, str, list, None]:
        pass