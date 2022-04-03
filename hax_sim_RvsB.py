from hax.classes.agents.ppo_agent import PPOAgent
from hax.classes.agents.random_agent import RandomAgent
from hax.classes.environments.simulation_environment import SimulationEnvironment
from hax.classes.models.sample_efficient_small import SampleEfficientSmallPPOModel
from hax.utils.supervisor import Supervisor

supervisor = Supervisor(
    environment=SimulationEnvironment(),
    teamLearned=SimulationEnvironment.Team.RED,
    redAgent=PPOAgent(
        model=SampleEfficientSmallPPOModel()
    ),
    blueAgent=RandomAgent(),
)

supervisor.run()
