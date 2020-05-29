"""
Run separate evaluation for comparing against BayRn
"""
import os
import os.path as osp
from datetime import datetime

from pyrado.algorithms.bayrn import BayRn
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.logger.experiment import ask_for_experiment, timestamp_format
from pyrado.utils.experiments import wrap_like_other_env, load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    # Create real-world counterpart (without domain randomization)
    env_real = QQubeReal(env_sim.dt, env_sim.max_steps)
    print_cbt(f'Set up the QQubeReal environment with dt={env_real.dt} max_steps={env_real.max_steps}.', 'c')
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run the policy on the real system
    ex_ts = datetime.now().strftime(timestamp_format)
    save_dir = osp.join(ex_dir, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    est_ret = BayRn.eval_policy(save_dir, env_real, policy, montecarlo_estimator=True, prefix=ex_ts, num_rollouts=5)

    print_cbt(f'Estimated return: {est_ret.item()}', 'g')
