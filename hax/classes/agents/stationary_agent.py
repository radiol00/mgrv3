from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment


class StationaryAgent(Agent):
    def chooseAction(self, state: Environment.State) -> Environment.Action:
        return Environment.Action.NO
