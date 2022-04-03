from typing import Any

from hax.classes.memory import Memory
from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


class PPOAgent(Agent):
    from tensorflow_probability.python.distributions import Categorical

    def __init__(self, model: PPOModel, memorySize: int = 120, name=None):
        super().__init__(name="PPO" if name is None else "PPO_"+name)
        self.model = model
        self.memory = Memory(size=memorySize)
        self.isTeachable = True

    def canLearn(self):
        return self.memory.newMemories >= self.memory.size

    def canForceLearn(self):
        return self.memory.newMemories >= self.model.batchSize

    def learn(self):
        actorLoss, criticLoss = self.model.learn(self.memory)
        self.model.usePlanner()
        self.memory.refresh()
        return actorLoss, criticLoss

    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any, Any):
        probs = self.model.actorPredict(state)
        val = self.model.criticPredict(state)
        distribution = self.Categorical(probs=probs)
        action_index = distribution.sample()
        return Environment.Action(action_index), distribution.log_prob(action_index), probs, val
