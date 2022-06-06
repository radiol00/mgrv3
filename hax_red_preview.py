from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.environments.image_extracting_env import ImageExtractingEnvironment
from hax.classes.models.sample_efficient_small import SampleEfficientSmallPPOModel
from hax.utils.argument_parser import ArgumentParser
from hax.utils.runner import Runner
from hax.utils.formatters import *

args = ArgumentParser()
env = ImageExtractingEnvironment(timeToLive=999_999_999)

redaldo = PPOAgent(
    model=SampleEfficientSmallPPOModel(
        actorWeightsPath=args.redActorWeights,
        criticWeightsPath=args.redCriticWeights,
        learningSessions=args.learningSessions,
        name="RED_PREVIEW"
    ),
    memorySize=0,
)

runner = Runner(dualCommand=False)
while runner.running:
    state = env.getState(bindState=True)
    actionRed, prob, val, probs = redaldo.chooseAction(state)
    env.doAction(actionRed, env.Action.NO)

    nextState = env.getState(bindState=False)
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
