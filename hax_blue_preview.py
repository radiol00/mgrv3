from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.environments.image_extracting_env import ImageExtractingEnvironment
from hax.classes.models.sample_efficient_small import SampleEfficientSmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.formatters import *

args = ArgumentParser()
env = ImageExtractingEnvironment(timeToLive=10 * 120)

bluessi = PPOAgent(
    model=SampleEfficientSmallPPOModel(
        actorWeightsPath=args.blueActorWeights,
        criticWeightsPath=args.blueCriticWeights,
        learningSessions=args.learningSessions,
        name="BLUE_PREVIEW"
    ),
    memorySize=24,
)

runner = Runner()
while runner.running:
    state = env.getState(keepLastState=True)
    actionBlue, probs, val = bluessi.chooseAction(state)
    env.doAction(actionBlue, env.Action.NO)

    nextState = env.getState(keepLastState=False)
    reward = env.getReward(nextState, env.Team.Blue)

    print(
        formatAPS(env.synchronizer.aps) + \
        formatActionProbabilities(probs) + \
        " " + \
        formatAction(actionBlue) + \
        " " + \
        formatReward(reward.value)
    )

runner.dispose()
env.dispose()
