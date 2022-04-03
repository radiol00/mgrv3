from hax.interfaces.environment import Environment
from haxballgym import make
from haxballgym.game import Game
import numpy as np


class SimulationEnvironment(Environment):

    @staticmethod
    def parseAction(action: Environment.Action):
        if action == Environment.Action.NO:
            return [1, 1, 0]
        elif action == Environment.Action.U:
            return [1, 0, 0]
        elif action == Environment.Action.D:
            return [1, 2, 0]
        elif action == Environment.Action.L:
            return [0, 1, 0]
        elif action == Environment.Action.R:
            return [2, 1, 0]
        elif action == Environment.Action.UL:
            return [0, 0, 0]
        elif action == Environment.Action.UR:
            return [2, 0, 0]
        elif action == Environment.Action.DL:
            return [0, 2, 0]
        elif action == Environment.Action.DR:
            return [2, 2, 0]
        elif action == Environment.Action.X_:
            return [1, 1, 1]
        elif action == Environment.Action.UX:
            return [1, 0, 1]
        elif action == Environment.Action.DX:
            return [1, 2, 1]
        elif action == Environment.Action.LX:
            return [0, 1, 1]
        elif action == Environment.Action.RX:
            return [2, 1, 1]
        elif action == Environment.Action.ULX:
            return [0, 0, 1]
        elif action == Environment.Action.URX:
            return [2, 0, 1]
        elif action == Environment.Action.DLX:
            return [0, 2, 1]
        elif action == Environment.Action.DRX:
            return [2, 2, 1]

    def __init__(self, timeToLive=120 * 5, stadiumFile="small.hbs"):
        super().__init__(timeToLive=timeToLive)
        self.game = make(game=Game(stadium_file=stadiumFile))
        self.game.reset()

    def getState(self, keepLastState: bool) -> Environment.State:
        red_pos = (np.array(self.game._match._game.players[0].disc.position) + [420, 205]).astype(int)
        blue_pos = (np.array(self.game._match._game.players[1].disc.position) + [420, 205]).astype(int)
        ball_pos = (np.array(self.game._match._game.stadium_game.discs[0].position) + [420, 205]).astype(int)
        rx, ry = red_pos
        bx, by = blue_pos
        ballx, bally = ball_pos

        if self.lastState is not None:
            last_red_pos, last_blue_pos, last_ball_pos = self.lastState
            drx, dry = (red_pos - last_red_pos)
            dbx, dby = (blue_pos - last_blue_pos)
            dballx, dbally = (ball_pos - last_ball_pos)
            state = Environment.State(
                ball=Environment.State.MapObject(x=ballx, y=bally, dx=dballx, dy=dbally),
                red=Environment.State.MapObject(x=rx, y=ry, dx=drx, dy=dry),
                blue=Environment.State.MapObject(x=bx, y=by, dx=dbx, dy=dby)
            )
        else:
            state = Environment.State(
                ball=Environment.State.MapObject(x=ballx, y=bally, dx=0, dy=0),
                red=Environment.State.MapObject(x=rx, y=ry, dx=0, dy=0),
                blue=Environment.State.MapObject(x=bx, y=by, dx=0, dy=0)
            )

        if keepLastState:
            self.lastState = red_pos, blue_pos, ball_pos
            self.age += 1
        return state

    def doAction(self, actionRed: Environment.Action, actionBlue: Environment.Action):
        self.game.step(actions=[self.parseAction(actionRed), self.parseAction(actionBlue)])

    def refresh(self):
        self.game.reset()
        self.forgetLastState()
        self.age = 0
        self.episodes += 1
