"""
Script to run the Quanser Car-Pole experiments reported in the journal paper
"""
import joblib
import numpy as np
import os.path as osp

import pyrado
from pyrado.environments.quanser.base import RealEnv
from pyrado.environments.quanser.quanser_cartpole import QCartPoleStabReal
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import ask_for_experiment, save_list_of_dicts_to_yaml, setup_experiment
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import wrap_like_other_env, load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


def volt_disturbance_pos(t: float):
    return [6.]


def volt_disturbance_neg(t: float):
    return [-6.]


def experiment_wo_distruber(env_real: RealEnv, env_sim: SimEnv):
    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt('Running the evaluation policy ...', 'c')
    return rollout(env_real, policy, eval=True, max_steps=args.max_steps, render_mode=RenderMode(text=True),
                   no_reset=True, no_close=True)


def experiment_w_distruber(env_real: RealEnv, env_sim: SimEnv):
    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt('Running the evaluation policy ...', 'c')
    ro1 = rollout(env_real, policy, eval=True, max_steps=args.max_steps//3, render_mode=RenderMode(),
                  no_reset=True, no_close=True)

    # Run disturber
    env_real = inner_env(env_real)  # since we are reusing it
    print_cbt('Running the 1st disturber ...', 'c')
    rollout(env_real, disturber_pos, eval=True, max_steps=steps_disturb, render_mode=RenderMode(),
            no_reset=True, no_close=True)

    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt('Running the evaluation policy ...', 'c')
    ro2 = rollout(env_real, policy, eval=True, max_steps=args.max_steps//3, render_mode=RenderMode(),
                  no_reset=True, no_close=True)

    # Run disturber
    env_real = inner_env(env_real)  # since we are reusing it
    print_cbt('Running the 2nd disturber ...', 'c')
    rollout(env_real, disturber_neg, eval=True, max_steps=steps_disturb, render_mode=RenderMode(),
            no_reset=True, no_close=True)

    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt('Running the evaluation policy ...', 'c')
    ro3 = rollout(env_real, policy, eval=True, max_steps=args.max_steps//3, render_mode=RenderMode(),
                  no_reset=True, no_close=True)

    return StepSequence.concat([ro1, ro2, ro3])


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()
    ex_tag = ex_dir.split('--', 1)[1]

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    if args.verbose:
        print(f'Policy params:\n{policy.param_values.detach().numpy()}')

    # Create real-world counterpart (without domain randomization)
    env_real = QCartPoleStabReal(args.dt, args.max_steps)
    print_cbt('Set up the QCartPoleStabReal environment.', 'c')

    # Set up the disturber
    disturber_pos = TimePolicy(env_real.spec, volt_disturbance_pos, env_real.dt)
    disturber_neg = TimePolicy(env_real.spec, volt_disturbance_neg, env_real.dt)
    steps_disturb = 10
    print_cbt(f'Set up the disturbers for the QCartPoleStabReal environment.'
              f'\nVolt disturbance: {6} volts for {steps_disturb} steps', 'c')

    # Center cart and reset velocity filters and wait until the user or the conroller has put pole upright
    env_real.reset()
    print_cbt('Ready', 'g')

    ros = []
    for r in range(args.num_runs):
        if args.mode == 'wo':
            ro = experiment_wo_distruber(env_real, env_sim)
        elif args.mode == 'w':
            ro = experiment_w_distruber(env_real, env_sim)
        else:
            raise pyrado.ValueErr(given=args.mode, eq_constraint="without (wo), or with (w) disturber")
        ros.append(ro)

    env_real.close()

    # Print and save results
    avg_return = np.mean([ro.undiscounted_return() for ro in ros])
    print_cbt(f'Average return: {avg_return}', 'g', bright=True)
    save_dir = setup_experiment('evaluation', 'qcp-st_experiment', ex_tag, base_dir=pyrado.TEMP_DIR)
    joblib.dump(ros, osp.join(save_dir, 'experiment_rollouts.pkl'))
    save_list_of_dicts_to_yaml(
        [dict(ex_dir=ex_dir, avg_return=avg_return, num_runs=len(ros), steps_disturb=steps_disturb)],
        save_dir, file_name='experiment_summary'
    )
