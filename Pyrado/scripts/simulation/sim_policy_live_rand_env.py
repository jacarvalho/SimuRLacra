"""
Simulate (with animation) a rollout in a live perturbed environment.
"""
import pyrado
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.utils import print_domain_params, get_default_randomizer
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Get the simulation environment
    env, policy, kwout = load_experiment(ex_dir)

    if not isinstance(env, DomainRandWrapperLive):
        # Add default domain randomization wrapper with action delay
        randomizer = get_default_randomizer(env)
        env = ActDelayWrapper(env)
        randomizer.add_domain_params(
            UniformDomainParam(name='act_delay', mean=5, halfspan=5, clip_lo=0, roundint=True))
        env = DomainRandWrapperLive(env, randomizer)
        print_cbt('Using default randomizer with additional action delay.', 'c')
    else:
        print_cbt('Using loaded randomizer.', 'c')

    # Simulate
    done, state, param = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=args.verbose, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=state))  # calls env.reset()
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, ro)
    pyrado.close_vpython()
