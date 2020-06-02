"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
import torch as to

import pyrado
from pyrado.algorithms.power import PoWER
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_wambic
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, f'{BayRn.name}_{PoWER.name}-sim2sim', 'cup_scale', seed=111)

    # Environments
    env_hparams = dict(max_steps=1000, task_args=dict(factor=0.2))
    env_sim = WAMBallInCupSim(**env_hparams)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_wambic()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = WAMBallInCupSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=7, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env_sim.spec, **policy_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=15,
        pop_size=35,
        num_rollouts=5,
        num_is_samples=15,
        expl_std_init=1.0,
        expl_std_min=0.05,
        num_sampler_envs=8,
    )
    power = PoWER(ex_dir, env_sim, policy, **subroutine_hparam)

    # Set the boundaries for the GP
    # .. cup scale: mean in (1.0, 1.5), halfspan in (0.1, 0.6)
    bounds = to.tensor(
        [[1.0, 0.1],
         [1.5, 0.6]]
    )

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc='EI',
        acq_restarts=50,
        acq_samples=500,
        num_init_cand=4,
        warmstart=False,
        num_eval_rollouts=10,
        thold_succ=500  # May need to be tunes...
    )
    # Save the environments and the hyper-parameters (do it before the init routine of BDR)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subroutine=subroutine_hparam, subroutine_name=PoWER.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subroutine=power, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
