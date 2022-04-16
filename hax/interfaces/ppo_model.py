import numpy as np
import os
import time

from hax.utils.memory import Memory
from hax.interfaces.environment import Environment


class PPOModel:
    from tensorflow.keras import Model
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    from tensorflow_probability.python.distributions import Categorical

    class Loss:
        def __init__(self, mean: float, minimum: float, maximum: float):
            self.mean = mean
            self.min = minimum
            self.max = maximum

    # HYPERPAREMETERS
    discountFactor = 0.99
    lambdaVal = 0.95

    batchSize = 4
    epochs = 16
    clip = 0.2

    lrA = 2e-6
    actorOptimizer = Adam(learning_rate=lrA)

    lrC = 5e-5
    criticOptimizer = Adam(learning_rate=lrC)
    # ----

    actor: Model
    critic: Model
    saveWeightsPerLS = 5
    stage = -1

    def __init__(self, actorWeightsPath, criticWeightsPath, learningSessions, name):
        self.name = name
        self.learningSessions = learningSessions
        self.actorWeightsPath = actorWeightsPath
        self.criticWeightsPath = criticWeightsPath
        self.stateShape = Environment.State.getShape()
        self.actionQuantity = Environment.Action.getQuantity()


        self.actor, self.critic = self.buildModels()

        self.loadWeights()
        self.usePlanner()

    def setActorLearningRate(self, val):
        self.lrA = val
        self.actorOptimizer = self.Adam(learning_rate=self.lrA)

    def setCriticLearningRate(self, val):
        self.lrC = val
        self.criticOptimizer = self.Adam(learning_rate=self.lrC)

    def setEpochs(self, val):
        self.epochs = val

    def setClip(self, val):
        self.clip = val

    def setBatchSize(self, val):
        self.batchSize = val

    def setLambdaVal(self, val):
        self.lambdaVal = val

    def setDiscountFactor(self, val):
        self.discountFactor = val

    def buildModels(self) -> (Model, Model):
        raise NotImplementedError

    def actorPredict(self, state: Environment.State):
        prediction = self.actor(np.array(state.toStateVector(normalized=True))[np.newaxis, :], training=False)[0]
        return prediction

    def criticPredict(self, state: Environment.State):
        prediction = self.critic(np.array(state.toStateVector(normalized=True))[np.newaxis, :], training=False)[0][0]
        return prediction

    def saveWeights(self):
        path = os.path.join("models", self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        t = time.time()
        self.actor.save_weights(os.path.join(path, self.name + f"-{self.learningSessions}-{t}-actor.hdf5"))
        self.critic.save_weights(os.path.join(path, self.name + f"-{self.learningSessions}-{t}-critic.hdf5"))
        print(f"{self.name} weights saved")

    def loadWeights(self):
        if self.criticWeightsPath is not None:
            self.critic.load_weights(self.criticWeightsPath)
            print(f"{self.name} critic weights loaded")

        if self.actorWeightsPath is not None:
            self.actor.load_weights(self.actorWeightsPath)
            print(f"{self.name} actor weights loaded")

    def calculateAdvantages(self, batch, memory: Memory):
        advantages = []
        norm_rewards = np.array(memory.rewards)
        norm_rewards = (norm_rewards - norm_rewards.mean()) / (norm_rewards.std() + 1e-10)
        for t in range(memory.memories - batch, memory.memories):

            advantage = 0
            reduction = 1

            for i in range(t, memory.memories - 1):
                reward = norm_rewards[i]
                val = memory.vals[i]
                next_val = memory.vals[i+1]
                advantage += reduction * ((reward + self.discountFactor * next_val) - val)
                reduction = reduction * self.discountFactor * self.lambdaVal

            if t == memory.memories - 1:
                advantage = norm_rewards[t]

            advantages.append(advantage)

        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages

    def learn(self, memory: Memory) -> (Loss, Loss):
        print(f"Learning a[{self.lrA}] c[{self.lrC}] epoch[{self.epochs}] batch[{self.batchSize}] discount[{self.discountFactor}] lambda[{self.lambdaVal}] clip[{self.clip}]")
        batchesQuan = memory.newMemories // self.batchSize
        lastMemories = batchesQuan * self.batchSize
        batch = np.arange(memory.memories - lastMemories, memory.memories)
        np.random.shuffle(batch)
        batches = [batch[i * self.batchSize:(i + 1) * self.batchSize] for i in range((len(batch) + self.batchSize - 1) // self.batchSize)]
        actorLosses = []
        criticLosses = []
        advantages = self.calculateAdvantages(lastMemories, memory)
        for batch in batches:
            states = memory.getStatesBatch(batch)
            actions = memory.getActionsBatch(batch)
            logProbs = self.tf.math.exp(memory.getLogProbsBatch(batch))
            vals = memory.getValsBatch(batch)
            adv = [advantages[i - (memory.memories - lastMemories)] for i in batch]
            adv = self.tf.convert_to_tensor(adv, dtype=self.tf.float32)

            for _ in range(self.epochs):
                with self.tf.GradientTape() as tape1, self.tf.GradientTape() as tape2:
                    probs = self.actor(states)
                    probs = self.tf.clip_by_value(probs, clip_value_min=1e-30, clip_value_max=1e+30)
                    newVals = self.critic(states)

                    dists = self.Categorical(probs=probs)
                    newLogProbs = self.tf.math.exp(dists.log_prob(actions))
                    probRatios = self.tf.divide(newLogProbs, logProbs)

                    advantageProbs = self.tf.multiply(adv, probRatios)
                    clippedAdvantageProbs = self.tf.multiply(self.tf.clip_by_value(probRatios, 1-self.clip, 1+self.clip), adv)
                    actorLoss = self.tf.multiply(self.tf.constant(-1.0), self.tf.reduce_mean(self.tf.minimum(advantageProbs, clippedAdvantageProbs)))
                    returns = self.tf.add(adv, vals)
                    criticLoss = self.tf.subtract(returns, newVals)
                    criticLoss = self.tf.square(criticLoss)
                    criticLoss = self.tf.reduce_mean(criticLoss)
                grad1 = tape1.gradient(actorLoss, self.actor.trainable_variables)
                grad2 = tape2.gradient(criticLoss, self.critic.trainable_variables)
                self.actorOptimizer.apply_gradients(zip(grad1, self.actor.trainable_variables))
                self.criticOptimizer.apply_gradients(zip(grad2, self.critic.trainable_variables))
                criticLosses.append(criticLoss.numpy())
                actorLosses.append(actorLoss.numpy())

        self.learningSessions += 1

        if self.learningSessions % self.saveWeightsPerLS == 0:
            self.saveWeights()

        actorLosses = np.array(actorLosses)
        criticLosses = np.array(criticLosses)
        return PPOModel.Loss(
            mean=actorLosses.mean(),
            minimum=actorLosses.min(initial=1e+5),
            maximum=actorLosses.max(initial=-1e+5)
        ), PPOModel.Loss(
            mean=criticLosses.mean(),
            minimum=criticLosses.min(initial=1e+5),
            maximum=criticLosses.max(initial=-1e+5)
        )

    def usePlanner(self):
        pass
