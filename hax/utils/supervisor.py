from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.memory import Memory
from hax.interfaces.agent import Agent
from hax.interfaces.environment import Environment
from hax.utils.argument_parser import ArgumentParser
from hax.utils.formatters import formatLosses
from hax.utils.runner import Runner
from hax.utils.statistics import Statistics


class Supervisor:
    def __init__(self, environment: Environment, teamLearned: Environment.Team, redAgent: Agent, blueAgent: Agent, muted=False, sampleSize=10_000):
        self.args = ArgumentParser()
        self.muted = muted
        self.environment = environment
        self.teamLearned = teamLearned
        self.redAgent = redAgent
        self.blueAgent = blueAgent
        statsName = ""
        if isinstance(self.redAgent, PPOAgent):
            statsName += self.redAgent.model.name
        else:
            statsName += self.redAgent.name

        statsName += "_VS_"

        if isinstance(self.blueAgent, PPOAgent):
            statsName += self.blueAgent.model.name
        else:
            statsName += self.blueAgent.name

        self.statistics = Statistics(sampleSize=sampleSize, statisticsSubDir=statsName)

        self.runner = Runner()

    def _getAgentToTeach(self) -> PPOAgent:
        if self.teamLearned == Environment.Team.RED and self.redAgent.isTeachable and isinstance(self.redAgent, PPOAgent):
            return self.redAgent
        if self.teamLearned == Environment.Team.BLUE and self.blueAgent.isTeachable and isinstance(self.blueAgent, PPOAgent):
            return self.blueAgent

        print("Selected team is not teachable!")
        exit(-1)

    def run(self):

        agent = self._getAgentToTeach()
        agent.model.setLearningSessions(self.args.learningSessions)

        if isinstance(self.redAgent, PPOAgent):
            self.redAgent.model.loadWeights(self.args.redCriticWeights, self.args.redActorWeights)

        if isinstance(self.blueAgent, PPOAgent):
            self.blueAgent.model.loadWeights(self.args.blueCriticWeights, self.args.blueActorWeights)

        while self.runner.running:
            state = self.environment.getState(keepLastState=True)

            redAction, redLogProb, redProbs, redVal = self.redAgent.chooseAction(state)
            blueAction, blueLogProb, blueProbs, blueVal = self.blueAgent.chooseAction(state)

            self.environment.doAction(redAction, blueAction)

            nextState = self.environment.getState(keepLastState=False)
            reward = self.environment.getReward(nextState, self.teamLearned)

            experience = Memory.Experience(
                normalized_state=state.toStateVector(normalized=True),
                state=state.toStateVector(normalized=False),
                reward=reward.value,
                rewardComponents=reward.components,
                done=reward.done,
                actionIndex=redAction.value if self.teamLearned == Environment.Team.RED else blueAction.value,
                val=redVal if self.teamLearned == Environment.Team.RED else blueVal,
                logProb=redLogProb if self.teamLearned == Environment.Team.RED else blueLogProb,
            )

            agentToTeach = self._getAgentToTeach()
            agentToTeach.memory.remember(experience)
            if reward.done:
                self.environment.refresh()
                if agentToTeach.canForceLearn():
                    actorLoss, criticLoss = agentToTeach.learn()
                    print(formatLosses(actorLoss, criticLoss))
                    self.statistics.addActorLoss(actorLoss)
                    self.statistics.addCriticLoss(criticLoss)
                else:
                    agentToTeach.memory.refresh()
            elif agentToTeach.canLearn():
                self.environment.onLearn()
                actorLoss, criticLoss = agentToTeach.learn()
                print(formatLosses(actorLoss, criticLoss))
                self.statistics.addActorLoss(actorLoss)
                self.statistics.addCriticLoss(criticLoss)
            elif self.environment.isOld():
                self.environment.refresh()
                if agentToTeach.canForceLearn():
                    actorLoss, criticLoss = agentToTeach.learn()
                    print(formatLosses(actorLoss, criticLoss))
                    self.statistics.addActorLoss(actorLoss)
                    self.statistics.addCriticLoss(criticLoss)
                else:
                    agentToTeach.memory.refresh()

        self.runner.dispose()
        self.environment.dispose()
        self.statistics.dump()
        self._getAgentToTeach().model.saveWeights()

