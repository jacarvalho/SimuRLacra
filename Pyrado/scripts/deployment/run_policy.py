"""
Run a policy (trained in simulation) on the associated real-world environment.
"""
import pyrado
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.utils import inner_env
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import wrap_like_other_env, load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    # Detect the correct real-world counterpart and create it
    if isinstance(inner_env(env_sim), QBallBalancerSim):
        env_real = QBallBalancerReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QCartPoleSim):
        env_real = QCartPoleReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QQubeSim):
        env_real = QQubeReal(dt=args.dt, max_steps=args.max_steps)
    else:
        raise pyrado.TypeErr(given=env_sim, expected_type=[QBallBalancerSim, QCartPoleSim, QQubeSim])
    print_cbt(f'Set up env {env_real.__class__}.', 'c')

    # Finally wrap the env in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    print_cbt('Running ...', 'c')
    while not done:
        ro = rollout(env_real, policy, eval=True, render_mode=RenderMode(text=False, video=args.animation))
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, _, _ = after_rollout_query(env_real, ro)
