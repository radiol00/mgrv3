from hax.utils.keyboard import KeyboardListener


class Runner:
    running = True
    __onHold = False

    def __init__(self, command="q", dualCommand=True):
        self.listener = KeyboardListener(onPressCallback=self.onPressListener)
        self.listener.start()
        self.command = command
        self.dualCommand = dualCommand

    def onPressListener(self, key):
        try:
            if key.char == self.command.lower() and self.running:
                if self.dualCommand:
                    self.__onHold = True
                else:
                    self.running = False
            elif key.char == self.command.upper() and self.running and self.__onHold:
                self.__onHold = False
                self.running = False
            else:
                self.__onHold = False
        except AttributeError:
            pass

    def dispose(self):
        self.listener.dispose()
