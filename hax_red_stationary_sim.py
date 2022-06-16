from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.agents.stationary_agent import StationaryAgent
from hax.classes.environments.simulation_environment import SimulationEnvironment
from hax.classes.models.small import SmallPPOModel
from hax.utils.memory import Memory
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.statistics import Statistics
from hax.utils.formatters import *

args = ArgumentParser()
env = SimulationEnvironment(timeToLive=10 * 120)

runName = "REDALDO_VS_STATIONARY_POLICY"

redaldo = PPOAgent(
    model=SmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions,
        name=runName
    ),
    memorySize=120,
)

bluessi = StationaryAgent()

stats = Statistics(
    sampleSize=10_000,
    statisticsSubDir=runName
)

if args.learningSessions == 0:
    redaldo.model.saveWeights()

runner = Runner(command="h")
while runner.running:
    state = env.getState(bindState=True)
    actionRed, prob, val, _ = redaldo.chooseAction(state)
    actionBlue = bluessi.chooseAction(state)
    env.doAction(actionRed, actionBlue)
    reward = env.getReward(env.getState(bindState=False), env.Team.Red)

    if reward.done == True:
        if reward.value < 0:
            print("LOSE")
        elif reward.value > 0:
            print("WIN")
        else:
            print("DRAW")

    experience = Memory.Experience(
        normalizedState=state.toStateVector(normalized=True),
        state=state.toStateVector(normalized=False),
        reward=reward.value,
        rewardComponents=reward.components,
        done=reward.done,
        actionIndex=actionRed.value,
        val=val,
        prob=prob,
    )

    print(
        formatLearningSessionInfo(newMemories=redaldo.memory.newMemories,
                                  memorySize=redaldo.memory.size,
                                  learningSessions=redaldo.model.learningSessions, ) + \
        " " + \
        formatEnvironmentCompletionInfo(envAge=env.age,
                                        envTimeToLive=env.timeToLive,
                                        envEpisodes=env.episodes) + \
        formatActionProbabilities(prob) + \
        " " + \
        formatAction(actionRed) + \
        " " + \
        formatReward(reward.value)
    )

    stats.addExperience(experience)
    redaldo.memory.remember(experience)

    actorLoss, criticLoss = redaldo.tryToLearn(experience, env)
    if actorLoss is not None and criticLoss is not None:
        print(formatLosses(actorLoss, criticLoss))
        stats.addActorLoss(actorLoss)
        stats.addCriticLoss(criticLoss)
        if redaldo.model.learningSessions >= 25_000:
            runner.running = False

runner.dispose()
env.dispose()
stats.dump()
redaldo.model.saveWeights()
