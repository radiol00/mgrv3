from typing import Any

from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel
import random


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def chooseAction(self, state: Environment.State) -> Environment.Action:
        return random.choice(list(Environment.Action))
