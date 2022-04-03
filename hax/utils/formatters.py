from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


def formatAPS(aps):
    if aps is None:
        return ""
    else:
        return f"[{aps}]"


def formatLearningSessionInfo(newMemories, memorySize, learningSessions):
    output = ""
    if newMemories is not None and memorySize is not None:
        learningSessionCompletionPercent = round(((newMemories + 1) / memorySize * 100), 1)
        output += f"LS[{learningSessionCompletionPercent}%]"
    if learningSessions is not None:
        output += f"[{learningSessions}]"
    return output


def formatEnvironmentCompletionInfo(envAge, envTimeToLive, envEpisodes):
    if envAge is None or envTimeToLive is None or envEpisodes is None:
        return ""

    environmentCompletionPercent = round((envAge / envTimeToLive) * 100, 1)
    return f"E[{environmentCompletionPercent}%][{envEpisodes}]"


def formatActionProbabilities(probs):
    probsString = ""
    offset = 9
    for i in range(9):
        probsString += f" {Environment.Action(i).name}"
        probsString += f"({probs[i]:.2f}|{probs[i + offset]:.2f})"

    return probsString


def formatAction(action):
    return f"[{action.name}]"


def formatReward(reward):
    return f"{round(reward, 3)}"


def formatLosses(actorLoss: PPOModel.Loss, criticLoss: PPOModel.Loss):
    return f"Actor Loss: mean({actorLoss.mean:.3f}) max({actorLoss.max:.3f}) min({actorLoss.min:.3f})\n" \
           f"Critic Loss: mean({criticLoss.mean:.3f}) max({criticLoss.max:.3f}) min({criticLoss.min:.3f})"
