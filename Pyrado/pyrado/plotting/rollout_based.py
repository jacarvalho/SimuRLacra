"""
Collection of plotting functions which take Rollouts as inputs and produce line plots
"""
import numpy as np
import functools
from matplotlib import gridspec
from matplotlib import pyplot as plt
from typing import Sequence

import pyrado
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.utils import remove_env
from pyrado.environments.base import Env
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import fill_list_of_arrays
from pyrado.utils.input_output import ensure_math_mode


def _get_obs_label(rollout: StepSequence, idx: int):
    try:
        label = f"{rollout.rollout_info['env_spec'].obs_space.labels[idx]}"
        if label == 'None':
            label = 'o_{' + f'{idx}' + '}'
    except (AttributeError, KeyError):
        label = 'o_{' + f'{idx}' + '}'
    return ensure_math_mode(label, no_subscript=True)


def _get_act_label(rollout: StepSequence, idx: int):
    try:
        label = f"{rollout.rollout_info['env_spec'].act_space.labels[idx]}"
        if label == 'None':
            label = 'a_{' + f'{idx}' + '}'
    except (AttributeError, KeyError):
        label = 'a_{' + f'{idx}' + '}'
    return ensure_math_mode(label, no_subscript=True)


def plot_observations_actions_rewards(ro: StepSequence):
    """
    Plot all observation, action, and reward trajectories of the given rollout (in different sub-plots).

    :param ro: input rollout
    """
    if hasattr(ro, 'observations') and hasattr(ro, 'actions') and hasattr(ro, 'env_infos'):
        dim_obs = ro.observations.shape[1]
        dim_act = ro.actions.shape[1]

        # Use recorded time stamps if possible
        t = ro.env_infos.get('t', np.arange(0, ro.length)) if hasattr(ro, 'env_infos') else np.arange(0, ro.length)

        fig, axs = plt.subplots(dim_obs + dim_act + 1, 1, figsize=(8, 12))
        fig.suptitle('Observations, Actions, and Reward over Time')
        plt.subplots_adjust(hspace=.5)

        # Observations (without the last time step)
        for i in range(dim_obs):
            axs[i].plot(t, ro.observations[:-1, i], label=_get_obs_label(ro, i), c=f'C{i%10}')
            axs[i].legend()

        # Actions
        for i in range(dim_act):
            axs[i + dim_obs].plot(t, ro.actions[:, i], label=_get_act_label(ro, i), c=f'C{i%10}')
            axs[i + dim_obs].legend()
        # action_labels = env.unwrapped.action_space.labels; label=action_labels[0]

        # Rewards
        axs[-1].plot(t, ro.rewards, label='reward')
        axs[-1].legend()

        plt.show()


def plot_observations(ro: StepSequence, idcs_sel: Sequence[int] = None):
    """
    Plot all observation trajectories of the given rollout (in individual sub-plots).

    :param ro: input rollout
    :param idcs_sel: indices of the selected selected observations, if `None` plot all
    """
    if hasattr(ro, 'observations'):
        # Select dimensions to plot
        dim_obs = range(ro.observations.shape[1]) if idcs_sel is None else idcs_sel

        # Use recorded time stamps if possible
        t = ro.env_infos.get('t', np.arange(0, ro.length)) if hasattr(ro, 'env_infos') else np.arange(0, ro.length)

        if len(dim_obs) <= 6:
            divisor = 2
        elif len(dim_obs) <= 12:
            divisor = 4
        else:
            divisor = 8
        num_cols = int(np.ceil(len(dim_obs)/divisor))
        num_rows = int(np.ceil(len(dim_obs)/num_cols))

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*5, num_rows*3))
        axs = axs.reshape((num_rows, num_cols))
        fig.suptitle('Observations over Time')
        plt.subplots_adjust(hspace=.5)

        if len(dim_obs) == 1:
            axs.plot(t, ro.observations[:-1, dim_obs[0]], label=_get_obs_label(ro, dim_obs[0]))
            axs.legend()
        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    if j + i*num_cols < len(dim_obs):
                        # Omit the last observation for simplicity
                        axs[i, j].plot(t, ro.observations[:-1, j + i*num_cols],
                                       label=_get_obs_label(ro, j + i*num_cols), c=f'C{i%10}')
                        axs[i, j].legend()
                    else:
                        # We might create more subplots than there are observations
                        pass

        plt.show()


def plot_actions(ro: StepSequence, env: Env = None):
    """
    Plot all action trajectories of the given rollout (in individual sub-plots).

    :param ro: input rollout
    :param env: environment (used for getting the clipped action values)
    """
    if hasattr(ro, 'actions'):
        dim_act = ro.actions.shape[1]
        # Use recorded time stamps if possible
        t = ro.env_infos.get('t', np.arange(0, ro.length)) if hasattr(ro, 'env_infos') else np.arange(0, ro.length)

        fig, axs = plt.subplots(dim_act, figsize=(8, 12))
        fig.suptitle('Actions over Time')
        plt.subplots_adjust(hspace=.5)

        if env is not None:
            act_space_unnorm = remove_env(env, ActNormWrapper).act_space
            act_clipped = np.array([act_space_unnorm.project_to(a) for a in ro.actions[:]])

        if dim_act == 1:
            axs.plot(t, ro.actions[:], label=_get_act_label(ro, 0))
            if env is not None:
                axs.plot(t, act_clipped, label=_get_act_label(ro, 0) + ' (clipped)', c='k', ls='--')
            axs.legend()
        else:

            for i in range(dim_act):
                axs[i].plot(t, ro.actions[:, i], label=_get_act_label(ro, i), c=f'C{i%10}')
                if env is not None:
                    axs[i].plot(t, act_clipped[:, i], label=_get_act_label(ro, i) + ' (clipped)', c='k', ls='--')
                axs[i].legend()
        plt.show()


def plot_rewards(ro: StepSequence):
    """
    Plot the reward trajectories of the given rollout.

    :param ro: input rollout
    """
    if hasattr(ro, 'rewards'):
        # Use recorded time stamps if possible
        t = ro.env_infos.get('t', np.arange(0, ro.length)) if hasattr(ro, 'env_infos') else np.arange(0, ro.length)

        fig, ax = plt.subplots(1)
        fig.suptitle('Reward over Time')
        ax.plot(t, ro.rewards)
        plt.show()


def plot_adn_data(ro: StepSequence):
    """
    Plot the trajectories specific to an Activation Dynamic Network (ADN).

    :param ro: input rollout
    """
    if hasattr(ro, 'actions') and hasattr(ro, 'potentials') and hasattr(ro, 'stimuli'):
        # Use recorded time stamps if possible
        t = ro.env_infos.get('t', np.arange(0, ro.length)) if hasattr(ro, 'env_infos') else np.arange(0, ro.length)
        num_rows = ro.potentials.shape[1]  # number of movement primitives / dynamical systems

        fig = plt.figure()
        gs = gridspec.GridSpec(num_rows, 3)

        for i in range(num_rows):
            ax0 = fig.add_subplot(gs[i, 0])
            ax0.plot(t, ro.stimuli[:, i], label=rf'$s_{i}$', c=f'C{i%10}')
            ax0.legend()
            ax1 = fig.add_subplot(gs[i, 1], sharex=ax0)
            ax1.plot(t, ro.potentials[:, i], label=rf'$p_{i}$', c=f'C{i%10}')
            ax1.legend()
            ax2 = fig.add_subplot(gs[i, 2], sharex=ax0)
            ax2.plot(t, ro.actions[:, i], label=rf'$a_{i}$', c=f'C{i%10}')
            ax2.legend()

            if i == 0:
                ax0.set_title('Stimuli over time')
                ax1.set_title('Potentials over time')
                ax2.set_title('Activations over time')

        plt.subplots_adjust(hspace=.2)
        plt.subplots_adjust(wspace=.3)
        plt.show()


def plot_statistic_across_rollouts(
    rollouts: Sequence[StepSequence],
    stat_fcn: callable,
    stat_fcn_kwargs=None,
    obs_idcs: Sequence[int] = None,
    act_idcs: Sequence[int] = None
):
    """
    Plot one statistic of interest (e.g. mean) across a list of rollouts.

    :param rollouts: list of rollouts, they can be of unequal length but are assumed to be from the same type of env
    :param stat_fcn: function to calculate the statistic of interest (e.g. np.mean)
    :param stat_fcn_kwargs: keyword arguments for the stat_fcn (e.g. {'axis': 0})
    :param obs_idcs: indices of the observations to process and plot
    :param act_idcs: indices of the actions to process and plot
    """
    if obs_idcs is None and act_idcs is None:
        raise pyrado.ValueErr(msg='Must select either an observation or an action, but both are None!')

    # Create figure with sub-figures
    num_subplts = 2 if (obs_idcs is not None and act_idcs is not None) else 1
    fix, axs = plt.subplots(num_subplts)

    if stat_fcn_kwargs is not None:
        stat_fcn = functools.partial(stat_fcn, **stat_fcn_kwargs)

    # Determine the longest rollout's length
    max_ro_len = max([ro.length for ro in rollouts])

    # Process observations
    if obs_idcs is not None:
        obs_sel = [ro.observations[:, obs_idcs] for ro in rollouts]
        obs_filled = fill_list_of_arrays(obs_sel, des_len=max_ro_len + 1)  # +1 since obs are of size ro.length+1
        obs_stat = stat_fcn(np.asarray(obs_filled))

        for i, obs_idx in enumerate(obs_idcs):
            axs[0].plot(obs_stat[:, i], label=_get_obs_label(rollouts[0], i), c=f'C{i%10}')
        axs[0].legend()

    # Process actions
    if act_idcs is not None:
        act_sel = [ro.actions[:, act_idcs] for ro in rollouts]
        act_filled = fill_list_of_arrays(act_sel, des_len=max_ro_len)
        act_stats = stat_fcn(np.asarray(act_filled))

        for i, act_idx in enumerate(act_idcs):
            axs[1].plot(act_stats[:, i], label=_get_act_label(rollouts[0], i), c=f'C{i%10}')
        axs[1].legend()

    plt.show()
