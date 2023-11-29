import random
import numpy as np
import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from .base import Strategy
from ..common import Actions
from typing import Callable

class LLMStrategy(Strategy):

    def __init__(
        self, 
        model: BaseChatModel,
        instructions: str,
        obs_preprocessor: Callable,
        name: str = None,
    ):
        super().__init__(name)

        self._model = model
        self._instructions = instructions
        self._obs_preprocessor = obs_preprocessor

        self._action_parser = RegexParser(
            regex=r"Action: (.*)", output_keys=["action"], default_output_key="action"
        )

        self._instructions

    def play(
        self, 
        observation: np.array, 
        reward: int = None
    ) -> Actions:
        obs = self._obs_preprocessor(observation)

        if reward is not None:
            self._return += reward 
        
        obs_message = f'''
        Observation: {obs}
        Reward: {reward}
        Return: {self._return}
        '''

        self._message_history.append(HumanMessage(content=obs_message))

        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_fixed(1),
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: print(
                    f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'
                ),
            ):
                with attempt:
                    act_message = self._model(self._message_history)
                    self._message_history.append(act_message)
                    action = int(self._action_parser.parse(act_message.content)["action"])
        
        except tenacity.RetryError as e:
            action = random.randint(Actions.C, Actions.D)

        return action

    def reset(self):
        super().reset()

        self._message_history = [
            SystemMessage(content=self._instructions)
        ]
        self._return = 0
