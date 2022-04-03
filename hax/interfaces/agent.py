from typing import Any
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.isTeachable = False

    def canLearn(self) -> bool:
        raise NotImplementedError

    def canForceLearn(self):
        raise NotImplementedError

    def learn(self) -> (PPOModel.Loss, PPOModel.Loss):
        raise NotImplementedError

    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any, Any):
        raise NotImplementedError
