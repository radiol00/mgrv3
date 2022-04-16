from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.environments.simulation_environment import SimulationEnvironment
from hax.utils.memory import Memory
from hax.classes.models.small import SmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.statistics import Statistics
from hax.utils.formatters import *

args = ArgumentParser()
env = SimulationEnvironment(timeToLive=10 * 120)
runName = "DOUBLE_MEM_BIGGER_LR_BLUESSI_VS_LEARNED_REDALDO_SMALL"

redaldo = PPOAgent(
    model=SmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions,
        name="LEARNED_REDALDO"
    ),
    memorySize=0,
)

bluessi = PPOAgent(
    model=SmallPPOModel(
        actorWeightsPath=args.blueActorWeights,
        criticWeightsPath=args.blueCriticWeights,
        learningSessions=args.learningSessions,
        name=runName
    ),
    memorySize=240,
)

stats = Statistics(
    sampleSize=10_000,
    statisticsSubDir=runName
)

if args.learningSessions == 0:
    bluessi.model.saveWeights()

runner = Runner(command="h")
while runner.running:
    state = env.getState(keepLastState=True)
    actionRed, _, _, _ = redaldo.chooseAction(state)
    actionBlue, probs, val = bluessi.chooseAction(state)
    env.doAction(actionRed, actionBlue)

    reward = env.getReward(env.getState(keepLastState=False), env.Team.Blue)

    # print(
    #     formatLearningSessionInfo(newMemories=bluessi.memory.newMemories,
    #                               memorySize=bluessi.memory.size,
    #                               learningSessions=bluessi.model.learningSessions, ) + \
    #     " " + \
    #     formatEnvironmentCompletionInfo(envAge=env.age,
    #                                     envTimeToLive=env.timeToLive,
    #                                     envEpisodes=env.episodes) + \
    #     formatActionProbabilities(probs) + \
    #     " " + \
    #     formatAction(actionBlue) + \
    #     " " + \
    #     formatReward(reward.value)
    # )

    experience = Memory.Experience(
        normalized_state=state.toStateVector(normalized=True),
        state=state.toStateVector(normalized=False),
        reward=reward.value,
        rewardComponents=reward.components,
        done=reward.done,
        actionIndex=actionBlue.value,
        val=val,
        prob=prob,
    )

    stats.addExperience(experience)

    actorLoss, criticLoss = bluessi.tryToLearn(experience, env)
    if actorLoss is not None and criticLoss is not None:
        print(formatLosses(actorLoss, criticLoss))
        stats.addActorLoss(actorLoss)
        stats.addCriticLoss(criticLoss)

runner.dispose()
env.dispose()
stats.dump()
bluessi.model.saveWeights()
