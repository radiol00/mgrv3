from hax.utils.keyboard import KeyboardListener


class Runner:
    running = True

    def __init__(self):
        self.listener = KeyboardListener(on_press_callback=self.onPressListener)
        self.listener.start()

    def onPressListener(self, key):
        try:
            if key.char == "q" and self.running:
                self.running = False
        except AttributeError:
            pass

    def dispose(self):
        self.listener.dispose()
