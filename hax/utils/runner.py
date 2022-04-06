from hax.utils.keyboard import KeyboardListener


class Runner:
    running = True
    __onHold = False

    def __init__(self, command="q"):
        self.listener = KeyboardListener(on_press_callback=self.onPressListener)
        self.listener.start()
        self.command = command

    def onPressListener(self, key):
        try:
            if key.char == self.command.lower() and self.running:
                self.__onHold = True
            elif key.char == self.command.upper() and self.running and self.__onHold:
                self.__onHold = False
                self.running = False
            else:
                self.__onHold = False
        except AttributeError:
            pass

    def dispose(self):
        self.listener.dispose()
