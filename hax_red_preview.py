from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.environments.image_extracting_env import ImageExtractingEnvironment
from hax.classes.models.sample_efficient_small import SampleEfficientSmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.formatters import *

args = ArgumentParser()
env = ImageExtractingEnvironment(timeToLive=10 * 120)

redaldo = PPOAgent(
    model=SampleEfficientSmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions,
        name="RED_PREVIEW"
    ),
    memorySize=24,
)

runner = Runner()
while runner.running:
    state = env.getState(keepLastState=True)
    actionRed, logProb, probs, val = redaldo.chooseAction(state)
    env.doAction(actionRed, env.Action.NO)

    nextState = env.getState(keepLastState=False)
    reward = env.getReward(nextState, env.Team.Red)

    print(
        formatAPS(env.synchronizer.aps) + \
        formatActionProbabilities(probs) + \
        " " + \
        formatAction(actionRed) + \
        " " + \
        formatReward(reward.value)
    )

runner.dispose()
env.dispose()
