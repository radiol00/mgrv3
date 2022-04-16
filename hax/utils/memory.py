class Memory:
    import tensorflow as tf
    from collections import deque

    class Experience:
        def __init__(self, normalized_state, state, reward, rewardComponents, done, actionIndex, val, prob):
            self.normalized_state = normalized_state
            self.state = state
            self.reward = reward
            self.rewardComponents = rewardComponents
            self.done = done
            self.actionIndex = actionIndex
            self.val = val
            self.prob = prob

    def __init__(self, size):
        self.states = self.deque(maxlen=size)
        self.rewards = self.deque(maxlen=size)
        self.dones = self.deque(maxlen=size)
        self.actionIndexes = self.deque(maxlen=size)
        self.probs = self.deque(maxlen=size)
        self.vals = self.deque(maxlen=size)
        self.size = size
        self.newMemories = 0
        self.memories = 0

    def remember(self, experience: Experience):
        self.states.append(experience.normalized_state)
        self.rewards.append(experience.reward)
        self.dones.append(experience.done)
        self.actionIndexes.append(experience.actionIndex)
        self.probs.append(experience.prob)
        self.vals.append(experience.val)

        self.newMemories += 1
        self.memories = len(self.states)

    def refresh(self):
        self.newMemories = 0

    def getStatesBatch(self, batch):
        states = [self.states[i] for i in batch]
        return self.tf.convert_to_tensor(states, dtype=self.tf.float32)

    def getRewardsBatch(self, batch):
        reward = [self.rewards[i] for i in batch]
        return self.tf.convert_to_tensor(reward, dtype=self.tf.float32)

    def getDonesBatch(self, batch):
        dones = [self.dones[i] for i in batch]
        return self.tf.convert_to_tensor(dones, dtype=self.tf.float32)

    def getProbsBatch(self, batch):
        probs = [self.probs[i] for i in batch]
        return self.tf.convert_to_tensor(probs, dtype=self.tf.float32)

    def getValsBatch(self, batch):
        vals = [self.vals[i] for i in batch]
        return self.tf.convert_to_tensor(vals, dtype=self.tf.float32)

    def getActionsBatch(self, batch):
        actions = [self.actionIndexes[i] for i in batch]
        return self.tf.convert_to_tensor(actions, dtype=self.tf.float32)
