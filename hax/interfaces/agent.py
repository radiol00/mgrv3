from typing import Any
from hax.interfaces.environment import Environment


class Agent:
    def __init__(self):
        self.isTeachable = False

    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any, Any):
        raise NotImplementedError
