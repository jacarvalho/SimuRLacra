"""
Simulate (with animation) a rollout in an environment.
"""
import pprint

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.environments.pysim.quanser_qube import QQubeSim, QQubeStabSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import RenderMode
import torch as to
import numpy as np


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Load the environment and the policy
    env = QQubeSim(args.dt, args.max_steps)  # runs infinitely by default

    # policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=0.04,  # Quanser's value: 0.02
    #     energy_gain=30.,  # Quanser's value: 50
    #     energy_th_gain=0.4,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-0.42, 18.45, -0.53, 1.53]))

    # QUANSER
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(np.log(0.02)),  # Quanser's value: 0.02
    #     energy_gain=np.exp(np.log(50.)),  # Quanser's value: 50
    #     energy_th_gain=0.3,  # former: 0.4
    #     # energy_th_gain=1.1970321,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-2., 35., -1.5, 3.]))

    # POWER
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(-3.4233),  # Quanser's value: 0.02
    #     energy_gain=np.exp( 4.1721),  # Quanser's value: 50
    #     energy_th_gain= 0.3,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-1.1408, 35.2608,  -1.1687, 3.7377]))

    # PGPE
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(-3.7156),  # Quanser's value: 0.02
    #     energy_gain=np.exp(3.9587),  # Quanser's value: 50
    #     energy_th_gain=0.3,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-2.1271, 34.8878, -1.5363, 2.8060]))

    # QUANSER with MVD energy controller for swing up
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(-2.6142373),  # Quanser's value: 0.02
    #     energy_gain=np.exp(2.6333313),  # Quanser's value: 50
    #     energy_th_gain=0.3,  # former: 0.4
    #     # energy_th_gain=1.1970321,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-2., 20., -1.0, 6.]))


    # Initial Controller
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(np.log(0.2)),  # Quanser's value: 0.02
    #     energy_gain=np.exp(np.log(50.)),  # Quanser's value: 50
    #     energy_th_gain=0.3,  # former: 0.4
    #     # energy_th_gain=1.1970321,  # former: 0.4
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-0.3, 9., -0.8, 2.]))

    # MVD
    # policy = QQubeSwingUpAndBalanceCtrl(
    #     env.spec,
    #     ref_energy=np.exp(-2.6142373),
    #     energy_gain=np.exp(2.6333313),
    #     energy_th_gain=0.3,  # for simulation and real system
    #     acc_max=5.,  # Quanser's value: 6
    #     alpha_max_pd_enable=10.,  # Quanser's value: 20
    #     pd_gains=to.tensor([-1.7313308, 35.976177, -1.58682, 3.0102878]))

    # Initial Controller SWINGUP ONLY
    policy = QQubeSwingUpAndBalanceCtrl(
        env.spec,
        ref_energy=np.exp(np.log(0.02)),  # Quanser's value: 0.02
        energy_gain=np.exp(np.log(50)),  # Quanser's value: 50
        energy_th_gain=0.3,
        acc_max=5.,
        alpha_max_pd_enable=10.,
        pd_gains=to.tensor([-0.42, 18.45, -0.53, 1.53]))


    print_cbt('Set up controller for the QQubeSim environment.', 'c')

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt


    if args.remove_dr_wrappers:
        env = remove_all_dr_wrappers(env, verbose=True)

    # Use the environments number of steps in case of the default argument (inf)
    max_steps = env.max_steps if args.max_steps == pyrado.inf else args.max_steps


    # Simulate
    done, state, param = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=args.verbose, video=args.animation),
                     eval=True, max_steps=max_steps, stop_on_done=not args.relentless,
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
    pyrado.close_vpython()
