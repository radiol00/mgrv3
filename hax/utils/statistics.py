import time
import os
import csv
from typing import List

from hax.classes.memory import Memory
from hax.interfaces.environment import Environment
from hax.interfaces.ppo_model import PPOModel


class Statistics:

    class ExperienceStat:
        def __init__(self, experience: Memory.Experience):
            self.value = experience
            self.timestamp = time.time()

    class LossStat:
        def __init__(self, loss: PPOModel.Loss):
            self.value = loss
            self.timestamp = time.time()

    def __init__(self, sampleSize, statisticsSubDir):
        self.sampleSize = sampleSize
        self.experienceMemory: List[Statistics.ExperienceStat] = []
        self.actorLossMemory: List[Statistics.LossStat] = []
        self.criticLossMemory: List[Statistics.LossStat] = []
        self.dumpIndex = 0
        self.timestamp = time.time()
        self.statisticsSubDir = statisticsSubDir
        self.path = os.path.join("statistics", self.statisticsSubDir, f"{self.timestamp}")

    def addExperience(self, experience):
        self.experienceMemory.append(Statistics.ExperienceStat(experience))
        if len(self.experienceMemory) >= self.sampleSize:
            self.dump()

    def addActorLoss(self, loss):
        self.actorLossMemory.append(Statistics.LossStat(loss))

    def addCriticLoss(self, loss):
        self.criticLossMemory.append(Statistics.LossStat(loss))

    def clearMemory(self):
        self.experienceMemory = []
        self.actorLossMemory = []
        self.criticLossMemory = []

    def dump(self):
        print(
            f"DUMPING STATISTICS:\n\t{len(self.experienceMemory)} - memories\n\t{len(self.actorLossMemory)} - actor losses\n\t{len(self.criticLossMemory)} - critic losses")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if len(self.experienceMemory) > 0:
            memoriesFieldnames = ["ball_x",
                                  "ball_y",
                                  "ball_v_x",
                                  "ball_v_y",
                                  "player_x",
                                  "player_y",
                                  "player_v_x",
                                  "player_v_y",
                                  "enemy_x",
                                  "enemy_y",
                                  "enemy_v_x",
                                  "enemy_v_y",
                                  "action",
                                  "reward",
                                  "done",
                                  "ts"]
            memoriesFieldnames.extend(Environment.getRewardComponentKeys())
            with open(os.path.join(self.path, f"{self.dumpIndex}-experiences.csv"), 'w', newline='') as memories_csvfile:
                writer = csv.DictWriter(memories_csvfile, fieldnames=memoriesFieldnames)
                writer.writeheader()
                for mem in self.experienceMemory:
                    ball_x, ball_y, ball_v_x, ball_v_y, player_x, player_y, player_v_x, player_v_y, enemy_x, enemy_y, enemy_v_x, enemy_v_y = mem.value.state
                    record = {
                        "ts": mem.timestamp,
                        "reward": round(mem.value.reward, 5),
                        "done": mem.value.done,
                        "action": Environment.Action(mem.value.actionIndex),
                        "ball_x": ball_x,
                        "ball_y": ball_y,
                        "ball_v_x": ball_v_x,
                        "ball_v_y": ball_v_y,
                        "player_x": player_x,
                        "player_y": player_y,
                        "player_v_x": player_v_x,
                        "player_v_y": player_v_y,
                        "enemy_x": enemy_x,
                        "enemy_y": enemy_y,
                        "enemy_v_x": enemy_v_x,
                        "enemy_v_y": enemy_v_y,
                    }
                    for key in Environment.getRewardComponentKeys():
                        record[key] = mem.value.rewardComponents[key]
                    writer.writerow(record)

        if len(self.actorLossMemory) > 0:
            with open(os.path.join(self.path, f"{self.dumpIndex}-actorLoss.csv"), 'w', newline='') as a_loss_csvfile:
                writer = csv.DictWriter(a_loss_csvfile, fieldnames=["loss", "max", "min", "ts"])
                writer.writeheader()
                for loss in self.actorLossMemory:
                    record = {
                        "loss": loss.value.mean,
                        "max": loss.value.max,
                        "min": loss.value.min,
                        "ts": loss.timestamp
                    }
                    writer.writerow(record)

        if len(self.criticLossMemory) > 0:
            with open(os.path.join(self.path, f"{self.dumpIndex}-criticLoss.csv"), 'w', newline='') as c_loss_csvfile:
                writer = csv.DictWriter(c_loss_csvfile, fieldnames=["loss", "max", "min", "ts"])
                writer.writeheader()
                for loss in self.criticLossMemory:
                    record = {
                        "loss": loss.value.mean,
                        "max": loss.value.max,
                        "min": loss.value.min,
                        "ts": loss.timestamp
                    }
                    writer.writerow(record)

        self.clearMemory()
        self.dumpIndex += 1
