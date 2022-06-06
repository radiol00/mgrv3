from enum import Enum
import numpy as np


class Environment:
    redGateX = 100
    redGateY = 205

    blueGateX = 740
    blueGateY = 205

    mapWidth = 840
    mapHeight = 410

    class Team(Enum):
        Red, Blue = range(2)

        def __str__(self):
            return self.name

    class Action(Enum):
        NO, U, D, L, R, UL, UR, DL, DR, X, UX, DX, LX, RX, ULX, URX, DLX, DRX = range(18)

        @staticmethod
        def getQuantity():
            return 18

    class State:
        class MapObject:
            class Normalized:
                def __init__(self, x: float, y: float, dx: float, dy: float):
                    self.x = x
                    self.y = y
                    self.dx = dx
                    self.dy = dy

            def __init__(self, x: int, y: int, dx: int, dy: int):
                self.x = x
                self.y = y
                self.dx = dx
                self.dy = dy

            def normalize(self) -> Normalized:
                return self.Normalized(
                    x=(self.x / Environment.mapWidth) - 0.5,
                    y=(self.y / Environment.mapHeight) - 0.5,
                    dx=(self.dx / Environment.mapWidth),
                    dy=(self.dy / Environment.mapHeight)
                )

        def __init__(self, red: MapObject, blue: MapObject, ball: MapObject):
            self.red = red
            self.blue = blue
            self.ball = ball

        @staticmethod
        def getShape():
            return 12

        def toStateVector(self, normalized=False):
            red = self.red.normalize() if normalized else self.red
            blue = self.blue.normalize() if normalized else self.blue
            ball = self.ball.normalize() if normalized else self.ball
            return [ball.x, ball.y, ball.dx, ball.dy, red.x, red.y, red.dx, red.dy, blue.x, blue.y, blue.dx, blue.dy]

    class Reward:
        drawReward = 0
        winReward = 200
        loseReward = -200

        def __init__(self, value: int, components: dict, done: bool):
            self.value = value
            self.components = components
            self.done = done

    @staticmethod
    def distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def getReward(self, state: State, team: Team) -> Reward:
        if self.isOld():
            return self.Reward(
                done=True,
                components={},
                value=self.Reward.drawReward
            )

        components = {}

        if team == Environment.Team.Red:
            enemyGateX, enemyGateY = self.blueGateX, self.blueGateY
            playerGateX, playerGateY = self.redGateX, self.redGateY
            enemy, player, ball = state.blue, state.red, state.ball
        else:
            enemyGateX, enemyGateY = self.redGateX, self.redGateY
            playerGateX, playerGateY = self.blueGateX, self.blueGateY
            enemy, player, ball = state.red, state.blue, state.ball

        # CHECK FOR GOAL
        if ball.x >= self.blueGateX:
            return self.Reward(
                value=self.Reward.winReward if team == Environment.Team.Red else self.Reward.loseReward,
                components=components,
                done=True
            )

        if ball.x <= self.redGateX:
            return self.Reward(
                value=self.Reward.winReward if team == Environment.Team.Blue else self.Reward.loseReward,
                components=components,
                done=True
            )

        components["ballGateTraj"] = 0

        if ball.dx != 0 or ball.dy != 0:
            enemyGateTrajectory = np.dot([ball.dx, ball.dy], [enemyGateX, enemyGateY])
            playerGateTrajectory = np.dot([ball.dx, ball.dy], [playerGateX, playerGateY])
            components["ballGateTraj"] = enemyGateTrajectory - playerGateTrajectory
            components["ballGateTraj"] *= 5e-4

        components["toBallTraj"] = 0

        if player.dx != 0 or player.dy != 0:
            d1 = self.distance(player.x, player.y, ball.x, ball.y)
            d2 = self.distance(player.x + (player.dx / 2),
                               player.y + (player.dy / 2),
                               ball.x,
                               ball.y)
            components["toBallTraj"] += d1 - d2

        if enemy.dx != 0 or enemy.dy != 0:
            d1 = self.distance(enemy.x, enemy.y, ball.x, ball.y)
            d2 = self.distance(enemy.x + (enemy.dx / 2),
                               enemy.y + (enemy.dy / 2),
                               ball.x,
                               ball.y)
            components["toBallTraj"] -= d1 - d2

        components["toBallTraj"] *= 8e-2

        reward = components["ballGateTraj"] + components["toBallTraj"]

        return self.Reward(
            value=reward,
            components=components,
            done=False
        )

    @staticmethod
    def getRewardComponentKeys():
        return ["ballGateTraj", "toBallTraj"]

    def __init__(self, timeToLive: int):
        self.lastState = None
        self.timeToLive = timeToLive
        self.age = 0
        self.episodes = 0

    def isOld(self):
        return self.age >= self.timeToLive

    def forgetLastState(self):
        self.lastState = None

    def refresh(self):
        self.forgetLastState()
        self.age = 0

    def getState(self, bindState: bool) -> State:
        raise NotImplementedError

    def doAction(self, actionRed: Action, actionBlue: Action):
        raise NotImplementedError

    def onLearn(self):
        pass

    def dispose(self):
        pass
