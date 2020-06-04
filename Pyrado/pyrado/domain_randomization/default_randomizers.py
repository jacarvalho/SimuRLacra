"""
Storage for default a.k.a. nominal domain parameter values and default randomizers
"""
import numpy as np
from typing import Dict, Tuple

from pyrado.domain_randomization.domain_parameter import BernoulliDomainParam, NormalDomainParam, UniformDomainParam
from pyrado.environments.sim_base import SimEnv
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer


default_randomizer_registry = {}


def default_randomizer(env_module, env_class):
    """
    Register a default randomizer provider for a given environment type.
    The environment type is referenced by name to avoid eager loading of all environments when this module is used.

    :usage:
    .. code-block:: python

        @default_randomizer('pyrado.environments.xy.my', 'MyEnv')
            def get_default_randomizer_my() -> DomainRandomizer:
                <implementation>
    
    :param env_module: Module in which the env class is defined.
    :param env_class: Environment class name.
    :return: decorator for default randomizer provider function.
    """

    def register(func):
        default_randomizer_registry[(env_module, env_class)] = func
        return func

    return register


def get_default_randomizer(env: [SimEnv, EnvWrapper]) -> DomainRandomizer:
    """
    Get the default randomizer depending on the passed environment.

    :param env: (wrapped) environment that should be perturbed
    :return: default randomizer
    """
    env_type = type(inner_env(env))

    # Try all env base types. This is more or less equivalent to isinstance
    for cand_type in env_type.__mro__:
        env_module = cand_type.__module__
        env_class = cand_type.__name__
        # Try to get it
        dp = default_randomizer_registry.get((env_module, env_class))
        if dp:
            return dp()
    else:
        # Not found
        raise ValueError(f'No default randomizer settings for env of type {env_type}!')


def get_conservative_randomizer(env: [SimEnv, EnvWrapper]) -> DomainRandomizer:
    """
    Get the default conservative randomizer depending on the passed environment.

    :param env: environment that should be perturbed
    :return: default conservative randomizer
    """
    randomizer = get_default_randomizer(env)
    randomizer.rescale_distr_param('std', 0.5)
    randomizer.rescale_distr_param('cov', np.sqrt(0.5))
    randomizer.rescale_distr_param('halfspan', 0.5)
    return randomizer


def get_zero_var_randomizer(env: [SimEnv, EnvWrapper]) -> DomainRandomizer:
    """
    Get the randomizer which always returns the nominal domain parameter values.

    :param env: environment that should be perturbed
    :return: randomizer with zero variance for all parameters
    """
    randomizer = get_default_randomizer(env)
    randomizer.rescale_distr_param('std', 0.)
    randomizer.rescale_distr_param('cov', 0.)
    randomizer.rescale_distr_param('halfspan', 0.)
    return randomizer


def get_empty_randomizer() -> DomainRandomizer:
    """
    Get an empty randomizer independent of the environment to be filled later (using `add_domain_params`).

    :return: empty randomizer
    """
    return DomainRandomizer()


def get_example_randomizer_cata() -> DomainRandomizer:
    """
    Get the randomizer for the `CatapultSim` used for the 'illustrative example' in F. Muratore et al, 2019, TAMPI.

    :return: randomizer based on the nominal domain parameter values
    """
    return DomainRandomizer(
        BernoulliDomainParam(name='planet', mean=None, val_0=0, val_1=1, prob_1=0.7, roundint=True)
    )  # 0 = Mars, 1 = Venus


@default_randomizer('pyrado.environments.one_step.catapult', 'CatapultSim')
def get_default_randomizer_cata() -> DomainRandomizer:
    """
    Get the default randomizer for the `CatapultSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.one_step.catapult import CatapultSim

    dp_nom = CatapultSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=dp_nom['g'], std=dp_nom['g']/5, clip_lo=1e-3),
        NormalDomainParam(name='k', mean=dp_nom['k'], std=dp_nom['k']/5, clip_lo=1e-3),
        NormalDomainParam(name='x', mean=dp_nom['x'], std=dp_nom['x']/5, clip_lo=1e-3)
    )


@default_randomizer('pyrado.environments.pysim.ball_on_beam', 'BallOnBeamSim')
def get_default_randomizer_bob() -> DomainRandomizer:
    """
    Get the default randomizer for the `BallOnBeamSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
    dp_nom = BallOnBeamSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=dp_nom['g'], std=dp_nom['g']/5, clip_lo=1e-4),
        NormalDomainParam(name='m_ball', mean=dp_nom['m_ball'], std=dp_nom['m_ball']/5, clip_lo=1e-4),
        NormalDomainParam(name='r_ball', mean=dp_nom['r_ball'], std=dp_nom['r_ball']/5, clip_lo=1e-4),
        NormalDomainParam(name='m_beam', mean=dp_nom['m_beam'], std=dp_nom['m_beam']/5, clip_lo=1e-3),
        NormalDomainParam(name='l_beam', mean=dp_nom['l_beam'], std=dp_nom['l_beam']/5, clip_lo=1e-3),
        NormalDomainParam(name='d_beam', mean=dp_nom['d_beam'], std=dp_nom['d_beam']/5, clip_lo=1e-3),
        UniformDomainParam(name='c_frict', mean=dp_nom['c_frict'], halfspan=dp_nom['c_frict'], clip_lo=0),
        UniformDomainParam(name='ang_offset', mean=0./180*np.pi, halfspan=0./180*np.pi)
    )


@default_randomizer('pyrado.environments.pysim.one_mass_oscillator', 'OneMassOscillatorSim')
def get_default_randomizer_omo() -> DomainRandomizer:
    """
    Get the default randomizer for the `OneMassOscillatorSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
    dp_nom = OneMassOscillatorSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='m', mean=dp_nom['m'], std=dp_nom['m']/3, clip_lo=1e-3),
        NormalDomainParam(name='k', mean=dp_nom['k'], std=dp_nom['k']/3, clip_lo=1e-3),
        NormalDomainParam(name='d', mean=dp_nom['d'], std=dp_nom['d']/3, clip_lo=1e-3)
    )


@default_randomizer('pyrado.environments.pysim.pendulum', 'PendulumSim')
def get_default_randomizer_pend() -> DomainRandomizer:
    """
    Get the default randomizer for the `PendulumSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.pendulum import PendulumSim
    dp_nom = PendulumSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=dp_nom['g'], std=dp_nom['g']/10, clip_lo=1e-3),
        NormalDomainParam(name='m_pole', mean=dp_nom['m_pole'], std=dp_nom['m_pole']/10, clip_lo=1e-3),
        NormalDomainParam(name='l_pole', mean=dp_nom['l_pole'], std=dp_nom['l_pole']/10, clip_lo=1e-3),
        NormalDomainParam(name='d_pole', mean=dp_nom['d_pole'], std=dp_nom['d_pole']/10, clip_lo=1e-3),
        NormalDomainParam(name='tau_max', mean=dp_nom['tau_max'], std=dp_nom['tau_max']/10, clip_lo=1e-3)
    )


@default_randomizer('pyrado.environments.pysim.quanser_ball_balancer', 'QBallBalancerSim')
def get_default_randomizer_qbb() -> DomainRandomizer:
    """
    Get the default randomizer for the `QBallBalancerSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
    dp_nom = QBallBalancerSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name=']', mean=dp_nom['g'], std=dp_nom['g']/5, clip_lo=1e-4),
        NormalDomainParam(name='m_ball', mean=dp_nom['m_ball'], std=dp_nom['m_ball']/5, clip_lo=1e-4),
        NormalDomainParam(name='r_ball', mean=dp_nom['r_ball'], std=dp_nom['r_ball']/5, clip_lo=1e-3),
        NormalDomainParam(name='l_plate', mean=dp_nom['l_plate'], std=dp_nom['l_plate']/5, clip_lo=5e-2),
        NormalDomainParam(name='r_arm', mean=dp_nom['r_arm'], std=dp_nom['r_arm']/5, clip_lo=1e-4),
        NormalDomainParam(name='K_g', mean=dp_nom['K_g'], std=dp_nom['K_g']/4, clip_lo=1e-2),
        NormalDomainParam(name='J_l', mean=dp_nom['J_l'], std=dp_nom['J_l']/4, clip_lo=1e-6),
        NormalDomainParam(name='J_m', mean=dp_nom['J_m'], std=dp_nom['J_m']/4, clip_lo=1e-9),
        NormalDomainParam(name='k_m', mean=dp_nom['k_m'], std=dp_nom['k_m']/4, clip_lo=1e-4),
        NormalDomainParam(name='R_m', mean=dp_nom['R_m'], std=dp_nom['R_m']/4, clip_lo=1e-4),
        UniformDomainParam(name='eta_g', mean=dp_nom['eta_g'], halfspan=dp_nom['eta_g']/4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name='eta_m', mean=dp_nom['eta_m'], halfspan=dp_nom['eta_m']/4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name='B_eq', mean=dp_nom['B_eq'], halfspan=dp_nom['B_eq']/4, clip_lo=1e-4),
        UniformDomainParam(name='c_frict', mean=dp_nom['c_frict'], halfspan=dp_nom['c_frict']/4, clip_lo=1e-4),
        UniformDomainParam(name='V_thold_x_pos', mean=dp_nom['V_thold_x_pos'], halfspan=dp_nom['V_thold_x_pos']/3),
        UniformDomainParam(name='V_thold_x_neg', mean=dp_nom['V_thold_x_neg'], halfspan=dp_nom['V_thold_x_neg']/3),
        UniformDomainParam(name='V_thold_y_pos', mean=dp_nom['V_thold_y_pos'], halfspan=dp_nom['V_thold_y_pos']/3),
        UniformDomainParam(name='V_thold_y_neg', mean=dp_nom['V_thold_y_neg'], halfspan=dp_nom['V_thold_y_neg']/3),
        UniformDomainParam(name='offset_th_x', mean=dp_nom['offset_th_x'], halfspan=6./180*np.pi),
        UniformDomainParam(name='offset_th_y', mean=dp_nom['offset_th_y'], halfspan=6./180*np.pi)
    )


@default_randomizer('pyrado.environments.pysim.quanser_cartpole', 'QCartPoleStabSim')
@default_randomizer('pyrado.environments.pysim.quanser_cartpole', 'QCartPoleSwingUpSim')
def get_default_randomizer_qcp() -> DomainRandomizer:
    """
    Get the default randomizer for the `QCartPoleSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
    dp_nom = QCartPoleSim.get_nominal_domain_param(long=False)
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=dp_nom['g'], std=dp_nom['g']/5, clip_lo=1e-4),
        NormalDomainParam(name='m_cart', mean=dp_nom['m_cart'], std=dp_nom['m_cart']/5, clip_lo=1e-4),
        NormalDomainParam(name='m_pole', mean=dp_nom['m_pole'], std=dp_nom['m_pole']/5, clip_lo=1e-4),
        NormalDomainParam(name='l_rail', mean=dp_nom['l_rail'], std=dp_nom['l_rail']/5, clip_lo=1e-2),
        NormalDomainParam(name='l_pole', mean=dp_nom['l_pole'], std=dp_nom['l_pole']/5, clip_lo=1e-2),
        UniformDomainParam(name='eta_m', mean=dp_nom['eta_m'], halfspan=dp_nom['eta_m']/4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name='eta_g', mean=dp_nom['eta_g'], halfspan=dp_nom['eta_g']/4, clip_lo=1e-4, clip_up=1),
        NormalDomainParam(name='K_g', mean=dp_nom['K_g'], std=dp_nom['K_g']/4, clip_lo=1e-4),
        NormalDomainParam(name='J_m', mean=dp_nom['J_m'], std=dp_nom['J_m']/4, clip_lo=1e-9),
        NormalDomainParam(name='r_mp', mean=dp_nom['r_mp'], std=dp_nom['r_mp']/5, clip_lo=1e-4),
        NormalDomainParam(name='R_m', mean=dp_nom['R_m'], std=dp_nom['R_m']/4, clip_lo=1e-4),
        NormalDomainParam(name='k_m', mean=dp_nom['k_m'], std=dp_nom['k_m']/4, clip_lo=1e-4),
        UniformDomainParam(name='B_eq', mean=dp_nom['B_eq'], halfspan=dp_nom['B_eq']/4, clip_lo=1e-4),
        UniformDomainParam(name='B_pole', mean=dp_nom['B_pole'], halfspan=dp_nom['B_pole']/4, clip_lo=1e-4)
    )


@default_randomizer('pyrado.environments.pysim.quanser_qube', 'QQubeSim')
def get_default_randomizer_qq() -> DomainRandomizer:
    """
    Get the default randomizer for the `QQubeSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.pysim.quanser_qube import QQubeSim
    dp_nom = QQubeSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='g', mean=dp_nom['g'], std=dp_nom['g']/5, clip_lo=1e-3),
        NormalDomainParam(name='Rm', mean=dp_nom['Rm'], std=dp_nom['Rm']/5, clip_lo=1e-3),
        NormalDomainParam(name='km', mean=dp_nom['km'], std=dp_nom['km']/5, clip_lo=1e-4),
        NormalDomainParam(name='Mr', mean=dp_nom['Mr'], std=dp_nom['Mr']/5, clip_lo=1e-4),
        NormalDomainParam(name='Lr', mean=dp_nom['Lr'], std=dp_nom['Lr']/5, clip_lo=1e-4),
        NormalDomainParam(name='Dr', mean=dp_nom['Dr'], std=dp_nom['Dr']/5, clip_lo=1e-9),
        NormalDomainParam(name='Mp', mean=dp_nom['Mp'], std=dp_nom['Mp']/5, clip_lo=1e-4),
        NormalDomainParam(name='Lp', mean=dp_nom['Lp'], std=dp_nom['Lp']/5, clip_lo=1e-4),
        NormalDomainParam(name='Dp', mean=dp_nom['Dp'], std=dp_nom['Dp']/5, clip_lo=1e-9)
    )


def get_uniform_masses_lengths_randomizer_qq(frac_halfspan: float):
    """
    Get a uniform randomizer that applies to all masses and lengths of the Quanser Qube according to a fraction of their
    nominal parameter values

    :param frac_halfspan: fraction of the nominal parameter value
    :return: `DomainRandomizer` with uniformly distributed masses and lengths
    """
    from pyrado.environments.pysim.quanser_qube import QQubeSim
    dp_nom = QQubeSim.get_nominal_domain_param()
    return DomainRandomizer(
        UniformDomainParam(name='Mp', mean=dp_nom['Mp'], halfspan=dp_nom['Mp']/frac_halfspan, clip_lo=1e-3),
        UniformDomainParam(name='Mr', mean=dp_nom['Mr'], halfspan=dp_nom['Mr']/frac_halfspan, clip_lo=1e-3),
        UniformDomainParam(name='Lr', mean=dp_nom['Lr'], halfspan=dp_nom['Lr']/frac_halfspan, clip_lo=1e-2),
        UniformDomainParam(name='Lp', mean=dp_nom['Lp'], halfspan=dp_nom['Lp']/frac_halfspan, clip_lo=1e-2),
    )


@default_randomizer('pyrado.environments.sim_rcs.ball_on_plate', 'BallOnPlateSim')
def get_default_randomizer_bop() -> DomainRandomizer:
    """
    Get the default randomizer for the `BallOnPlateSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.ball_on_plate import BallOnPlateSim
    dp_nom = BallOnPlateSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='ball_mass', mean=dp_nom['ball_mass'], std=dp_nom['ball_mass']/3, clip_lo=1e-2),
        NormalDomainParam(name='ball_radius', mean=dp_nom['ball_radius'], std=dp_nom['ball_radius']/3, clip_lo=1e-2),
        NormalDomainParam(name='ball_com_x', mean=dp_nom['ball_com_x'], std=0.003),
        NormalDomainParam(name='ball_com_y', mean=dp_nom['ball_com_y'], std=0.003),
        NormalDomainParam(name='ball_com_z', mean=dp_nom['ball_com_z'], std=0.003),
        UniformDomainParam(name='ball_friction_coefficient', mean=dp_nom['ball_friction_coefficient'],
                           halfspan=dp_nom['ball_friction_coefficient'], clip_lo=0, clip_hi=1),
        UniformDomainParam(name='ball_rolling_friction_coefficient', mean=dp_nom['ball_rolling_friction_coefficient'],
                           halfspan=dp_nom['ball_rolling_friction_coefficient'], clip_lo=0, clip_hi=1),
        # Vortex only
        UniformDomainParam(name='ball_slip', mean=dp_nom['ball_slip'], halfspan=dp_nom['ball_slip'], clip_lo=0)
        # UniformDomainParam(name='ball_linearvelocitydamnping', mean=0., halfspan=0.),
        # UniformDomainParam(name='ball_angularvelocitydamnping', mean=0., halfspan=0.)
    )


@default_randomizer('pyrado.environments.sim_rcs.planar_insert', 'PlanarInsertSim')
def get_default_randomizer_pi() -> DomainRandomizer:
    """
    Get the default randomizer for the `PlanarInsertSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim
    dp_nom = PlanarInsertSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='link1_mass', mean=dp_nom['link1_mass'], std=dp_nom['link1_mass']/5, clip_lo=1e-2),
        NormalDomainParam(name='link2_mass', mean=dp_nom['link2_mass'], std=dp_nom['link2_mass']/5, clip_lo=1e-2),
        NormalDomainParam(name='link3_mass', mean=dp_nom['link3_mass'], std=dp_nom['link3_mass']/5, clip_lo=1e-2),
        NormalDomainParam(name='link4_mass', mean=dp_nom['link4_mass'], std=dp_nom['link4_mass']/5, clip_lo=1e-2),
        NormalDomainParam(name='link5_mass', mean=dp_nom['link4_mass'], std=dp_nom['link4_mass']/5, clip_lo=1e-2),
        UniformDomainParam(name='upperwall_pos_offset_z', mean=0, halfspan=0.05, clip_lo=0)  # only increase the gap
    )


@default_randomizer('pyrado.environments.sim_rcs.box_shelving', 'BoxShelvingPosMPsSim')
@default_randomizer('pyrado.environments.sim_rcs.box_shelving', 'BoxShelvingVelMPsSim')
def get_default_randomizer_bs() -> DomainRandomizer:
    """
    Get the default randomizer for the `BoxShelvingSim`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.box_shelving import BoxShelvingSim
    dp_nom = BoxShelvingSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='box_length', mean=dp_nom['box_length'], std=dp_nom['box_length']/10),
        NormalDomainParam(name='box_width', mean=dp_nom['box_width'], std=dp_nom['box_width']/10),
        NormalDomainParam(name='box_mass', mean=dp_nom['box_mass'], std=dp_nom['box_mass']/5),
        UniformDomainParam(name='box_friction_coefficient', mean=dp_nom['box_friction_coefficient'],
                           halfspan=dp_nom['box_friction_coefficient']/5, clip_lo=1e-5),
    )


@default_randomizer('pyrado.environments.sim_rcs.box_lifting', 'BoxLiftingPosMPsSim')
@default_randomizer('pyrado.environments.sim_rcs.box_lifting', 'BoxLiftingVelMPsSim')
@default_randomizer('pyrado.environments.sim_rcs.box_lifting', 'BoxLiftingSimplePosMPsSim')
@default_randomizer('pyrado.environments.sim_rcs.box_lifting', 'BoxLiftingSimpleVelMPsSim')
def get_default_randomizer_bl() -> DomainRandomizer:
    """
    Get the default randomizer for the `BoxLifting`.

    :return: randomizer based on the nominal domain parameter values
    """
    from pyrado.environments.rcspysim.box_shelving import BoxShelvingSim
    dp_nom = BoxShelvingSim.get_nominal_domain_param()
    return DomainRandomizer(
        NormalDomainParam(name='box_length', mean=dp_nom['box_length'], std=dp_nom['box_length']/10),
        NormalDomainParam(name='box_width', mean=dp_nom['box_width'], std=dp_nom['box_width']/10),
        NormalDomainParam(name='box_mass', mean=dp_nom['box_mass'], std=dp_nom['box_mass']/5),
        UniformDomainParam(name='box_friction_coefficient', mean=dp_nom['box_friction_coefficient'],
                           halfspan=dp_nom['box_friction_coefficient']/5, clip_lo=1e-5),
        NormalDomainParam(name='basket_mass', mean=dp_nom['basket_mass'], std=dp_nom['basket_mass']/5),
        UniformDomainParam(name='basket_friction_coefficient', mean=dp_nom['basket_friction_coefficient'],
                           halfspan=dp_nom['basket_friction_coefficient']/5, clip_lo=1e-5),
    )


@default_randomizer('pyrado.environments.mujoco.wam', 'WAMBallInCupSim')
def get_default_randomizer_wambic() -> DomainRandomizer:
    return DomainRandomizer(
        UniformDomainParam(name='cup_scale', mean=1.3, halfspan=0.5)
    )


def get_default_domain_param_map_omo() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters as used in the `BayRn` algorithm.

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ('m', 'mean'),
        1: ('m', 'std'),
        2: ('k', 'mean'),
        3: ('k', 'std'),
        4: ('k', 'mean'),
        5: ('d', 'std'),
        6: ('d', 'mean'),
    }


def get_default_domain_param_map_pend() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters as used in the `BayRn` algorithm.

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ('tau_max', 'mean'),
        1: ('tau_max', 'std'),
    }


def get_default_domain_param_map_qq() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters as used in the `BayRn` algorithm.

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ('Mp', 'mean'),
        1: ('Mp', 'std'),
        2: ('Mr', 'mean'),
        3: ('Mr', 'std'),
        4: ('Lp', 'mean'),
        5: ('Lp', 'std'),
        6: ('Lr', 'mean'),
        7: ('Lr', 'std'),
    }


def get_default_domain_param_map_wambic() -> Dict[int, Tuple[str, str]]:
    """
    Get the default mapping from indices to domain parameters as used in the `BayRn` algorithm.

    :return: `dict` where the key is the index and the value is a tuple of domain parameter and the associated domain
             distribution parameter
    """
    return {
        0: ('cup_scale', 'mean'),
        1: ('cup_scale', 'halfspan'),
    }
