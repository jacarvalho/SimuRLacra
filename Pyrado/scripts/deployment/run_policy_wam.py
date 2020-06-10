"""
Run a policy (trained in simulation) on real Barret WAM.
"""
import pyrado
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.barrett_wam.wam import WAMBallInCupReal
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
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
    if isinstance(inner_env(env_sim), WAMBallInCupSim):
        # If `max_steps` (or `dt`) are not explicitly set using `args`, use the same as in the simulation
        max_steps = args.max_steps if args.max_steps < pyrado.inf else env_sim.max_steps
        dt = args.dt if args.dt is not None else env_sim.dt
        env_real = WAMBallInCupReal(dt=dt, max_steps=max_steps)
    else:
        raise pyrado.TypeErr(given=env_sim, expected_type=WAMBallInCupSim)

    # Finally wrap the env in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    while not done:
        ro = rollout(env_real, policy, eval=True)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, _, _ = after_rollout_query(env_real, policy, ro)
