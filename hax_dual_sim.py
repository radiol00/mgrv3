from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.agents.random_agent import RandomAgent
from hax.classes.environments.simulation_environment import SimulationEnvironment
from hax.utils.memory import Memory
from hax.classes.models.small import SmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.statistics import Statistics
from hax.utils.formatters import *

args = ArgumentParser()
env = SimulationEnvironment(timeToLive=10 * 120)

runName = "DUAL_SIM_MEDIUM_RELU"

redaldo = PPOAgent(
    model=SmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions,
        name=runName + "_RED",
    ),
    memorySize=120,
)

bluessi = PPOAgent(
    model=SmallPPOModel(
        actorWeightsPath=args.blueActorWeights,
        criticWeightsPath=args.blueCriticWeights,
        learningSessions=args.learningSessions,
        name=runName + "_BLUE",
    ),
    memorySize=120,
)

statsRed = Statistics(
    sampleSize=25_000,
    statisticsSubDir=runName + "_RED"
)

statsBlue = Statistics(
    sampleSize=25_000,
    statisticsSubDir=runName + "_BLUE"
)

if args.learningSessions == 0:
    redaldo.model.saveWeights()
    bluessi.model.saveWeights()

runner = Runner(command="h")
while runner.running:
    state = env.getState(bindState=True)
    actionRed, probRed, valRed, _ = redaldo.chooseAction(state)
    actionBlue, probBlue, valBlue, _ = bluessi.chooseAction(state)
    env.doAction(actionRed, actionBlue)
    rewardRed = env.getReward(env.getState(bindState=False), env.Team.Red)
    rewardBlue = env.getReward(env.getState(bindState=False), env.Team.Blue)

    if rewardRed.done == True and rewardBlue.done == True:
        if rewardRed.value > 0:
            print("RED WIN")
        elif rewardBlue.value > 0:
            print("BLUE WIN")
        else:
            print("DRAW")

    experienceRed = Memory.Experience(
        normalizedState=state.toStateVector(normalized=True),
        state=state.toStateVector(normalized=False),
        reward=rewardRed.value,
        rewardComponents=rewardRed.components,
        done=rewardRed.done,
        actionIndex=actionRed.value,
        val=valRed,
        prob=probRed,
    )

    experienceBlue = Memory.Experience(
        normalizedState=state.toStateVector(normalized=True),
        state=state.toStateVector(normalized=False),
        reward=rewardBlue.value,
        rewardComponents=rewardBlue.components,
        done=rewardBlue.done,
        actionIndex=actionBlue.value,
        val=valBlue,
        prob=probBlue,
    )


    statsRed.addExperience(experienceRed)
    statsBlue.addExperience(experienceBlue)

    redaldo.memory.remember(experienceRed)
    bluessi.memory.remember(experienceBlue)

    if experienceRed.done and experienceBlue.done:
        env.refresh()
        if redaldo.canForceLearn() and bluessi.canForceLearn():
            env.onLearn()
            actorLossRed, criticLossRed = redaldo.learn()
            actorLossBlue, criticLossBlue = bluessi.learn()
            statsRed.addActorLoss(actorLossRed)
            statsRed.addCriticLoss(criticLossRed)
            statsBlue.addActorLoss(actorLossBlue)
            statsBlue.addCriticLoss(criticLossBlue)
        else:
            redaldo.memory.refresh()
            bluessi.memory.refresh()
    elif redaldo.canLearn() and bluessi.canLearn():
            env.onLearn()
            actorLossRed, criticLossRed = redaldo.learn()
            actorLossBlue, criticLossBlue = bluessi.learn()
            statsRed.addActorLoss(actorLossRed)
            statsRed.addCriticLoss(criticLossRed)
            statsBlue.addActorLoss(actorLossBlue)
            statsBlue.addCriticLoss(criticLossBlue)
    elif env.isOld():
        env.refresh()
        if redaldo.canForceLearn() and bluessi.canForceLearn():
            env.onLearn()
            actorLossRed, criticLossRed = redaldo.learn()
            actorLossBlue, criticLossBlue = bluessi.learn()
            statsRed.addActorLoss(actorLossRed)
            statsRed.addCriticLoss(criticLossRed)
            statsBlue.addActorLoss(actorLossBlue)
            statsBlue.addCriticLoss(criticLossBlue)
        else:
            redaldo.memory.refresh()
            bluessi.memory.refresh()

    if redaldo.model.learningSessions >= 15_000 and bluessi.model.learningSessions >= 15_000:
        runner.running = False

runner.dispose()
env.dispose()
statsRed.dump()
statsBlue.dump()
redaldo.model.saveWeights()
bluessi.model.saveWeights()
