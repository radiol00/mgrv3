import concurrent.futures
import platform

import cv2
import mss
import numpy as np
from PIL import Image

from hax.interfaces.environment import Environment
from hax.utils.keyboard import KeyboardController
from hax.utils.synchronizer import Synchronizer


class ImageExtractingEnvironment(Environment):
    def __init__(self, timeToLive=120 * 5, aps=4):
        super().__init__(timeToLive=timeToLive)
        if platform.system() == "Darwin":
            self.left = 147
            self.top = 381
        else:
            self.left = 60
            self.top = 327
        self.screenRecorder = mss.mss()

        ballFile = "hax/resources/pics/ball.png"
        self.ball = cv2.imread(ballFile)

        redFile = "hax/resources/pics/red.png"
        self.red = cv2.imread(redFile)

        blueFile = "hax/resources/pics/blue.png"
        self.blue = cv2.imread(blueFile)

        self.controller = KeyboardController()

        self.synchronizer = Synchronizer(apsLimit=aps)
        self.synchronizer.sync()

    def __getStateFromFrame(self, frame, bindState=True) -> Environment.State:
        def threadedFinder(args):
            img, lastPos, threadFrame = args

            def findNormally():
                result = cv2.matchTemplate(threadFrame, img, cv2.TM_SQDIFF_NORMED)

                mn, _, mnLoc, _ = cv2.minMaxLoc(result)
                x, y = mnLoc
                x += int(img.shape[0] / 2)
                y += int(img.shape[1] / 2) + 1
                return x, y

            def getFramePortion(f, x, y, size):
                # PADDING
                padded = cv2.copyMakeBorder(f, size, size, size, size, cv2.BORDER_CONSTANT)

                return padded[y:y+2*size, x: x+2*size]

            if lastPos is None:
                return findNormally()
            else:
                lastX, lastY = lastPos
                portionSize = 100
                patternThreshold = 0.1
                portion = getFramePortion(threadFrame, lastX, lastY, portionSize)
                portionResult = cv2.matchTemplate(portion, img, cv2.TM_SQDIFF_NORMED)
                pmn, _, pmnLoc, _ = cv2.minMaxLoc(portionResult)

                if pmn > patternThreshold:
                    return findNormally()
                portionX, portionY = pmnLoc
                portionX += int(img.shape[0] / 2)
                portionY += int(img.shape[1] / 2) + 1

                globalX = lastX - portionSize + portionX
                globalY = lastY - portionSize + portionY
                return globalX, globalY

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self.lastState is None:
                params = [(self.ball, None, frame), (self.red, None, frame), (self.blue, None, frame)]
            else:
                bx, by, Rx, Ry, Bx, By = self.lastState
                params = [(self.ball, (bx, by), frame), (self.red, (Rx, Ry), frame), (self.blue, (Bx, By), frame)]

            futures = [executor.submit(threadedFinder, param) for param in params]
            returns = [f.result() for f in futures]

        ballX, ballY = returns[0]
        redX, redY = returns[1]
        blueX, blueY = returns[2]

        if self.lastState is None:
            ballVX = 0
            ballVY = 0

            redVX = 0
            redVY = 0

            blueVX = 0
            blueVY = 0
        else:
            lastBallX, lastBallY, lastPlayerX, lastPlayerY, lastEnemyX, lastEnemyY = self.lastState

            ballVX = ballX - lastBallX
            ballVY = ballY - lastBallY

            redVX = redX - lastPlayerX
            redVY = redY - lastPlayerY

            blueVX = blueX - lastEnemyX
            blueVY = blueY - lastEnemyY

        if bindState:
            self.lastState = ballX, ballY, redX, redY, blueX, blueY
            self.age += 1

        return Environment.State(
            ball=Environment.State.MapObject(ballX, ballY, ballVX, ballVY),
            red=Environment.State.MapObject(redX, redY, redVX, redVY),
            blue=Environment.State.MapObject(blueX, blueY, blueVX, blueVY)
        )

    def getMonitor(self):
        if platform.system() == 'Darwin':
            monitor = {'top': self.top, 'left': self.left, 'width': Environment.mapWidth//2, 'height': Environment.mapHeight//2}
        else:
            monitor = {'top': self.top, 'left': self.left, 'width': Environment.mapWidth, 'height': Environment.mapHeight}
        return monitor

    def __getFrame(self) -> np.array:
        ss = self.screenRecorder.grab(self.getMonitor())
        # noinspection PyTypeChecker
        return np.array(ss, dtype="uint8")[:, :, :3]

    def saveTestFrame(self, path="hax_frame.png"):
        ss = self.screenRecorder.grab(self.getMonitor())
        Image.frombytes("RGB", (ss.width, ss.height), ss.rgb).save(path)
        print(f"Test Frame saved to {path}")

    def getState(self, bindState: bool) -> Environment.State:
        frame = self.__getFrame()
        return self.__getStateFromFrame(frame, bindState=bindState)

    def doAction(self, action: Environment.Action, _: Environment.Action):
        self.controller.doAction(action)
        self.synchronizer.sync()

    def dispose(self):
        self.controller.release()

    def onLearn(self):
        self.controller.release()

    def refresh(self):
        self.controller.release()
        self.controller.refreshCombo()
        self.lastState = None
        self.age = 0
        self.episodes += 1
