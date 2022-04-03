import time

from hax.interfaces.environment import Environment


class KeyboardListener:
    from pynput.keyboard import Listener

    def __init__(self, on_press_callback=lambda: None):
        self.listener = self.Listener(on_press=on_press_callback)
        self.listener_alive = False

    def start(self):
        self.listener.start()
        self.listener_alive = True

    def dispose(self):
        self.listener.stop()
        self.listener_alive = False


class KeyboardController:
    import platform
    listener = None
    listener_alive = False
    isDarwin = platform.system() == "Darwin"

    if isDarwin:
        import pyautogui
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.MINIMUM_SLEEP = 0
        pyautogui.PAUSE = 0
    else:
        import pydirectinput
        pydirectinput.MINIMUM_DURATION = 0
        pydirectinput.MINIMUM_SLEEP = 0
        pydirectinput.PAUSE = 0

    def keyDown(self, key):
        if self.isDarwin:
            self.pyautogui.keyDown(key)
        else:
            self.pydirectinput.keyDown(key)

    def keyUp(self, key):
        if self.isDarwin:
            self.pyautogui.keyUp(key)
        else:
            self.pydirectinput.keyUp(key)

    def press(self, key):
        if self.isDarwin:
            self.pyautogui.press(key)
        else:
            self.pydirectinput.press(key)

    def refreshCombo(self):
        self.press("enter")
        time.sleep(0.3)
        self.press("r")
        time.sleep(0.3)
        self.press("enter")
        time.sleep(0.3)

    def doAction(self, action):
        self.release()
        if action == Environment.Action.U:  # up
            self.keyDown("up")
        elif action == Environment.Action.D:  # down
            self.keyDown("down")
        elif action == Environment.Action.L:  # left
            self.keyDown("left")
        elif action == Environment.Action.R:  # right
            self.keyDown("right")
        elif action == Environment.Action.UL:
            self.keyDown("up")
            self.keyDown("left")
        elif action == Environment.Action.UR:
            self.keyDown("up")
            self.keyDown("right")
        elif action == Environment.Action.DL:
            self.keyDown("down")
            self.keyDown("left")
        elif action == Environment.Action.DR:
            self.keyDown("down")
            self.keyDown("right")
        elif action == Environment.Action.UX:  # up
            self.keyDown("up")
            self.keyDown("x")
        elif action == Environment.Action.DX:  # down
            self.keyDown("down")
            self.keyDown("x")
        elif action == Environment.Action.LX:  # left
            self.keyDown("left")
            self.keyDown("x")
        elif action == Environment.Action.RX:  # right
            self.keyDown("right")
            self.keyDown("x")
        elif action == Environment.Action.X:  # x
            self.keyDown("x")
        elif action == Environment.Action.ULX:
            self.keyDown("up")
            self.keyDown("left")
            self.keyDown("x")
        elif action == Environment.Action.URX:
            self.keyDown("up")
            self.keyDown("right")
            self.keyDown("x")
        elif action == Environment.Action.DLX:
            self.keyDown("down")
            self.keyDown("left")
            self.keyDown("x")
        elif action == Environment.Action.DRX:
            self.keyDown("down")
            self.keyDown("right")
            self.keyDown("x")

    def release(self):
        for button in ["up", "down", "left", "right", "x"]:
            self.keyUp(button)
