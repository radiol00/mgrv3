from tensorflow.keras import Model

from hax.interfaces.ppo_model import PPOModel


class SampleEfficientSmallPPOModel(PPOModel):
    def __init__(self):
        super().__init__()
        self.name = "SAMPLE_EFFICIENT_SMALL"
        self.saveWeightsPerLS = 100

        self.setBatchSize(8)
        self.setActorLearningRate(1e-5)
        self.setCriticLearningRate(1e-4)
        self.setEpochs(32)

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

    def usePlanner(self):
        pass
