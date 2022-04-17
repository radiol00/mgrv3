from tensorflow.keras import Model

from hax.interfaces.ppo_model import PPOModel


class SmallPPOModel(PPOModel):
    def __init__(self, actorWeightsPath, criticWeightsPath, learningSessions, name):
        super().__init__(actorWeightsPath, criticWeightsPath, learningSessions, name)

        self.saveWeightsPerLS = 100

    def buildModels(self) -> (Model, Model):
        from tensorflow.keras.layers import Input, Dense
        AinputL = Input(shape=(self.stateShape,))
        AhiddenL1 = Dense(64, activation="tanh")(AinputL)
        AhiddenL2 = Dense(64, activation="tanh")(AhiddenL1)
        AprobabilityL = Dense(self.actionQuantity, activation="softmax")(AhiddenL2)

        CinputL = Input(shape=(self.stateShape,))
        ChiddenL1 = Dense(64, activation="tanh")(CinputL)
        ChiddenL2 = Dense(64, activation="tanh")(ChiddenL1)
        CqValueL = Dense(1, activation=None)(ChiddenL2)

        actor = Model([AinputL], [AprobabilityL])
        critic = Model([CinputL], [CqValueL])

        return actor, critic
        
    # planner for red vs random
    def usePlanner(self):
        if self.stage == -1:
            self.setBatchSize(8)
            self.setActorLearningRate(1e-5)
            self.setCriticLearningRate(1e-4)
            self.setEpochs(8)
            self.stage = 0

        if self.stage == 0 and self.learningSessions >= 370000:
            self.setActorLearningRate(5e-6)
            self.setCriticLearningRate(5e-5)
            self.setBatchSize(4)
            self.stage = 1
