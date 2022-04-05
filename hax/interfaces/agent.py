from typing import Any
from hax.interfaces.environment import Environment


class Agent:
    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any, Any):
        raise NotImplementedError
