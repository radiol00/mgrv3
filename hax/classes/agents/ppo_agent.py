from typing import Any

from hax.utils.memory import Memory
from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


class PPOAgent(Agent):
    from tensorflow_probability.python.distributions import Categorical

    def __init__(self, model: PPOModel, memorySize: int = 120):
        super().__init__()
        self.model = model
        self.memory = Memory(size=memorySize)

    def canLearn(self):
        return self.memory.newMemories >= self.memory.size

    def canForceLearn(self):
        return self.memory.newMemories >= self.model.batchSize

    def tryToLearn(self, experience: Memory.Experience, environment: Environment):
        self.memory.remember(experience)

        if experience.done:
            environment.refresh()
            if self.canForceLearn():
                environment.onLearn()
                actorLoss, criticLoss = self.learn()
                return actorLoss, criticLoss
            else:
                self.memory.refresh()
        elif self.canLearn():
            environment.onLearn()
            actorLoss, criticLoss = self.learn()
            return actorLoss, criticLoss
        elif environment.isOld():
            environment.refresh()
            if self.canForceLearn():
                environment.onLearn()
                actorLoss, criticLoss = self.learn()
                return actorLoss, criticLoss
            else:
                self.memory.refresh()
        return None, None

    def learn(self):
        actorLoss, criticLoss = self.model.learn(self.memory)
        self.model.usePlanner()
        self.memory.refresh()
        return actorLoss, criticLoss

    def chooseAction(self, state: Environment.State) -> (Environment.Action, Any, Any):
        probs = self.model.actorPredict(state)
        val = self.model.criticPredict(state)
        distribution = self.Categorical(probs=probs)
        actionIndex = distribution.sample()
        return Environment.Action(actionIndex), distribution.prob(actionIndex), val.numpy()
