from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
import random


class RandomAgent(Agent):
    def chooseAction(self, state: Environment.State) -> Environment.Action:
        return random.choice(list(Environment.Action))
