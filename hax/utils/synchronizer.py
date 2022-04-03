class Synchronizer:
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    import time

    def __init__(self, apsLimit=4):
        self.clock = self.pygame.time.Clock()
        self.lastTick = None
        self.aps = "APS"
        self.apsLimit = apsLimit

    def sync(self, tick=True):
        self.apsCalculate()
        self.lastTick = self.time.time()
        if tick:
            self.clock.tick(self.apsLimit)

    def apsCalculate(self):
        if self.lastTick is None:
            return

        d = (self.time.time() - self.lastTick)
        if d == 0:
            self.aps = "Max"
        else:
            self.aps = round(1 / d, 1)
