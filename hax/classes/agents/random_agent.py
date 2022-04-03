from typing import Any

from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel
import random


class RandomAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name="RANDOM" if name is None else "RANDOM_"+name)

    def canLearn(self) -> bool:
        return False

    def canForceLearn(self):
        return False

    def learn(self) -> (PPOModel.Loss, PPOModel.Loss):
        raise NotImplementedError

    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any, Any):
        return random.choice(list(Environment.Action)), None, None, None
