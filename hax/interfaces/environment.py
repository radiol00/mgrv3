from enum import Enum
import numpy as np


class Environment:
    redGateX = 100
    redGateY = 205

    blueGateX = 740
    blueGateY = 205

    class Team(Enum):
        RED, BLUE = range(2)

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
                    x=(self.x / 840) - 0.5,
                    y=(self.y / 410) - 0.5,
                    dx=(self.dx / 840),
                    dy=(self.dy / 410)
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
        draw_reward = 0
        win_reward = 200
        lose_reward = -200

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
                value=self.Reward.draw_reward
            )

        components = {}

        if team == Environment.Team.RED:
            enemyGateX, enemyGateY = self.blueGateX, self.blueGateY
            playerGateX, playerGateY = self.redGateX, self.redGateY
            enemy, player, ball = state.blue, state.red, state.ball
        else:
            enemyGateX, enemyGateY = self.redGateX, self.redGateY
            playerGateX, playerGateY = self.blueGateX, self.blueGateY
            enemy, player, ball = state.red, state.blue, state.ball

        # CHECK FOR GOAL
        if state.ball.x >= self.blueGateX:
            return self.Reward(
                value=self.Reward.win_reward if team == Environment.Team.RED else self.Reward.lose_reward,
                components=components,
                done=True
            )

        if state.ball.x <= self.redGateX:
            return self.Reward(
                value=self.Reward.win_reward if team == Environment.Team.BLUE else self.Reward.lose_reward,
                components=components,
                done=True
            )

        components["BallGateTraj"] = 0

        if ball.dx != 0 or ball.dy != 0:
            enemyGateTrajectory = np.dot([ball.dx, ball.dy], [enemyGateX, enemyGateY])
            playerGateTrajectory = np.dot([ball.dx, ball.dy], [playerGateX, playerGateY])
            components["BallGateTraj"] = enemyGateTrajectory - playerGateTrajectory
            components["BallGateTraj"] *= 5e-4

        components["PlayerBallTraj"] = 0

        if player.dx != 0 or player.dy != 0:
            d1 = self.distance(player.x, player.y, ball.x, ball.y)
            d2 = self.distance(player.x + (player.dx / 2),
                               player.y + (player.dy / 2),
                               ball.x,
                               ball.y)
            components["PlayerBallTraj"] += d1 - d2

        if enemy.dx != 0 or enemy.dy != 0:
            d1 = self.distance(enemy.x, enemy.y, ball.x, ball.y)
            d2 = self.distance(enemy.x + (enemy.dx / 2),
                               enemy.y + (enemy.dy / 2),
                               ball.x,
                               ball.y)
            components["PlayerBallTraj"] -= d1 - d2

        components["PlayerBallTraj"] *= 8e-2

        reward = components["BallGateTraj"] + components["PlayerBallTraj"]

        return self.Reward(
            value=reward,
            components=components,
            done=False
        )

    @staticmethod
    def getRewardComponentKeys():
        return ["PlayerBallTraj", "BallGateTraj"]

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

    def getState(self, keepLastState: bool) -> State:
        raise NotImplementedError

    def doAction(self, actionRed: Action, actionBlue: Action):
        raise NotImplementedError

    def onLearn(self):
        pass

    def dispose(self):
        pass
