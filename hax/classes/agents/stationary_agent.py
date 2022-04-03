from typing import Any

from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


class StationaryAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name="STATIONARY" if name is None else "STATIONARY_"+name)

    def canLearn(self) -> bool:
        return False

    def canForceLearn(self):
        return False

    def learn(self) -> (PPOModel.Loss, PPOModel.Loss):
        raise NotImplementedError

    def chooseAction(self, state: Environment.State) -> Environment.Action:
        return Environment.Action.NO
