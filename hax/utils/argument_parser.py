import argparse


class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-rc', '--red_critic',
                            dest='red_critic_weights', default=None,
                            help="path to red critic network weights")

        parser.add_argument('-ra', '--red_actor',
                            dest='red_actor_weights', default=None,
                            help="path to red actor network weights")

        parser.add_argument('-bc', '--blue_critic',
                            dest='blue_critic_weights', default=None,
                            help="path to blue critic network weights")

        parser.add_argument('-ba', '--blue_actor',
                            dest='blue_actor_weights', default=None,
                            help="path to blue actor network weights")

        parser.add_argument('-tf', '--test_frame',
                            dest='test_frame', default=False, action="store_true",
                            help="if set, program only saves test frame to be sure what model will see")

        parser.add_argument('-ls',
                            dest='learning_sessions', type=int, default=0,
                            help="# of completed learning sessions, this value is used by planner to choose correct learning stage")

        parser.add_argument('-m',
                            dest="muted", default=False, action="store_true",
                            help="if set, program will try to print as as little as possible")

        args = parser.parse_args()

        self.redCriticWeights = args.red_critic_weights
        self.redActorWeights = args.red_actor_weights

        self.blueCriticWeights = args.blue_critic_weights
        self.blueActorWeights = args.blue_actor_weights

        self.testFrame = args.test_frame

        self.learningSessions = args.learning_sessions

        self.muted = args.muted
