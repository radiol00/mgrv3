from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.agents.random_agent import RandomAgent
from hax.classes.environments.simulation_environment import SimulationEnvironment
from hax.classes.memory import Memory
from hax.classes.models.sample_efficient_small import SampleEfficientSmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.statistics import Statistics
from hax.utils.formatters import *

args = ArgumentParser()
env = SimulationEnvironment(timeToLive=10 * 120)

redaldo = PPOAgent(
    name="REDALDO",
    model=SampleEfficientSmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions
    ),
    memorySize=24,
)

bluessi = RandomAgent(
    name="BLUESSI"
)

stats = Statistics(
    sampleSize=10_000,
    statisticsSubDir=redaldo.model.name
)

if args.learningSessions == 0:
    redaldo.model.saveWeights()

runner = Runner()
while runner.running:
    state = env.getState(keepLastState=True)
    actionRed, logProb, probs, val = redaldo.chooseAction(state)
    actionBlue = bluessi.chooseAction(state)
    env.doAction(actionRed, actionBlue)

    reward = env.getReward(env.getState(keepLastState=False), env.Team.RED)

    print(
        formatLearningSessionInfo(newMemories=redaldo.memory.newMemories,
                                  memorySize=redaldo.memory.size,
                                  learningSessions=redaldo.model.learningSessions, ) + \
        " " + \
        formatEnvironmentCompletionInfo(envAge=env.age,
                                        envTimeToLive=env.timeToLive,
                                        envEpisodes=env.episodes) + \
        formatActionProbabilities(probs) + \
        " " + \
        formatAction(actionRed) + \
        " " + \
        formatReward(reward.value)
    )

    experience = Memory.Experience(
        normalized_state=state.toStateVector(normalized=True),
        state=state.toStateVector(normalized=False),
        reward=reward.value,
        rewardComponents=reward.components,
        done=reward.done,
        actionIndex=actionRed.value,
        val=val,
        logProb=logProb,
    )

    stats.addExperience(experience)

    actorLoss, criticLoss = redaldo.tryToLearn(experience, env)
    if actorLoss is not None and criticLoss is not None:
        print(formatLosses(actorLoss, criticLoss))
        stats.addActorLoss(actorLoss)
        stats.addCriticLoss(criticLoss)

runner.dispose()
env.dispose()
stats.dump()
redaldo.model.saveWeights()
