import numpy as np
import time
import torch as to
import torch.nn as nn
from numpy.random import binomial
from tabulate import tabulate

import pyrado
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.base import Env
from pyrado.environments.quanser.base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.environment_wrappers.utils import inner_env, typed_env
from pyrado.plotting.curve import plot_dts
from pyrado.plotting.policy_parameters import render_policy_params
from pyrado.plotting.rollout_based import plot_observations_actions_rewards, plot_actions, plot_observations, \
    plot_rewards, plot_potentials, plot_features
from pyrado.policies.adn import ADNPolicy
from pyrado.policies.base import Policy
from pyrado.policies.neural_fields import NFPolicy
from pyrado.policies.two_headed import TwoHeadedPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt, color_validity


def rollout(env: Env,
            policy: [nn.Module, Policy],
            eval: bool = False,
            max_steps: int = None,
            reset_kwargs: dict = None,
            render_mode: RenderMode = RenderMode(),
            render_step: int = 1,
            bernoulli_reset: float = None,
            no_reset: bool = False,
            no_close: bool = False,
            record_dts: bool = False,
            stop_on_done: bool = True) -> StepSequence:
    """
    Perform a rollout (i.e. sample a trajectory) in the given environment using given policy.

    :param env: environment to use (`SimEnv` or `RealEnv`)
    :param policy: policy to determine the next action given the current observation.
                   This policy may be wrapped by an exploration strategy.
    :param eval: flag if the rollout is executed during training (`False`) or during evaluation (`True`)
    :param max_steps: maximum number of time steps, if `None` the environment's property is used
    :param reset_kwargs: keyword arguments passed to environment's reset function
    :param render_mode: determines if the user sees an animation, console prints, or nothing
    :param render_step: rendering interval, renders every step if set to 1
    :param bernoulli_reset: probability for resetting after the current time step
    :param no_reset: do not reset the environment before running the rollout
    :param no_close: do not close (and disconnect) the environment after running the rollout
    :param record_dts: flag if the time intervals of different parts of one step should be recorded (for debugging)
    :param stop_on_done: set to false to ignore the environments's done flag (for debugging)
    :return paths of the observations, actions, rewards, and information about the environment as well as the policy
    """
    # Check the input
    if not isinstance(env, Env):
        raise pyrado.TypeErr(given=env, expected_type=Env)
    # Don't restrain policy type, can be any callable
    if not isinstance(eval, bool):
        raise pyrado.TypeErr(given=eval, expected_type=bool)
    # The max_steps argument is checked by the environment's setter
    if not (isinstance(reset_kwargs, dict) or reset_kwargs is None):
        raise pyrado.TypeErr(given=reset_kwargs, expected_type=dict)
    if not isinstance(render_mode, RenderMode):
        raise pyrado.TypeErr(given=render_mode, expected_type=RenderMode)

    # Initialize the paths
    obs_hist = []
    act_hist = []
    rew_hist = []
    env_info_hist = []
    if policy.is_recurrent:
        hidden_hist = []
    # If an ExplStrat is passed use the policy property, if a Policy is passed use it directly
    if isinstance(getattr(policy, 'policy', policy), (ADNPolicy, NFPolicy)):
        pot_hist = []
        stim_ext_hist = []
        stim_int_hist = []
    elif isinstance(getattr(policy, 'policy', policy), TwoHeadedPolicy):
        head_2_hist = []
    if record_dts:
        dt_policy_hist = []
        dt_step_hist = []
        dt_remainder_hist = []

    # Override the number of steps to execute
    if max_steps is not None:
        env.max_steps = max_steps

    # Reset the environment and pass the kwargs
    if reset_kwargs is None:
        reset_kwargs = {}
    if not no_reset:
        obs = env.reset(**reset_kwargs)
    else:
        obs = np.zeros(env.obs_space.shape)

    if isinstance(policy, Policy):
        # Reset the policy / the exploration strategy
        policy.reset()

        # Set dropout and batch normalization layers to the right mode
        if eval:
            policy.eval()
        else:
            policy.train()

    # Check for recurrent policy, which requires special handling
    if policy.is_recurrent:
        # Initialize hidden state var
        hidden = policy.init_hidden()

    # Setup rollout information
    rollout_info = dict(env_spec=env.spec)
    if isinstance(inner_env(env), SimEnv):
        rollout_info['domain_param'] = env.domain_param

    # Initialize animation
    env.render(render_mode, render_step=1)

    # Initialize the main loop variables
    done = False
    if record_dts:
        t_post_step = time.time()  # first sample of remainder is useless

    # ----------
    # Begin loop
    # ----------

    # Terminate if the environment signals done, it also keeps track of the time
    while not (done and stop_on_done) and env.curr_step < env.max_steps:
        # Record step start time
        if record_dts or render_mode.video:
            t_start = time.time()  # dual purpose
        if record_dts:
            dt_remainder = t_start - t_post_step

        # Check observations
        if np.isnan(obs).any():
            env.render(render_mode, render_step=1)
            raise pyrado.ValueErr(
                msg=f'At least one observation value is NaN!' +
                    tabulate([list(env.obs_space.labels),
                              [*color_validity(obs, np.invert(np.isnan(obs)))]], headers='firstrow')
            )

        # Get the agent's action
        obs_to = to.from_numpy(obs).type(to.get_default_dtype())  # policy operates on PyTorch tensors
        with to.no_grad():
            if policy.is_recurrent:
                if isinstance(getattr(policy, 'policy', policy), TwoHeadedPolicy):
                    act_to, head_2_to, hidden_next = policy(obs_to, hidden)
                else:
                    act_to, hidden_next = policy(obs_to, hidden)
            else:
                if isinstance(getattr(policy, 'policy', policy), TwoHeadedPolicy):
                    act_to, head_2_to = policy(obs_to)
                else:
                    act_to = policy(obs_to)

                    # act_to = (to.tensor([-3.6915228, 31.47042,   -6.827999,  11.602707]) @ obs_to).view(-1)


                    # act_to = (to.tensor([-0.42, 18.45, -0.53, 1.53]) @ obs_to).view(-1)
                    # act_to = (to.tensor([-0.2551887, 9.8527975, -4.421094, 10.82632]) @ obs_to).view(-1)



                    # act_to = (to.tensor([ 0.18273291 , 3.829101 ,  -1.4158,      5.5001416]) @ obs_to).view(-1)


                    # act_to = to.tensor([1.0078554 , 4.221323 ,  0.032006 ,  4.909644,  -2.201612]) @ obs_to

                    # act_to = to.tensor([1.89549804,  4.74797034, -0.09684278,  5.51203606, -2.80852473]) @ obs_to

                    # act_to = to.tensor([1.3555347 ,  3.8478632,  -0.04043245 , 7.40247 ,   -3.580207]) @ obs_to + \
                    #     0.1 * np.random.randn()

                    # print(act_to)


        act = act_to.detach().cpu().numpy()  # environment operates on numpy arrays

        # Check actions
        if np.isnan(act).any():
            env.render(render_mode, render_step=1)
            raise pyrado.ValueErr(
                msg=f'At least one observation value is NaN!' +
                    tabulate([list(env.act_space.labels),
                              [*color_validity(act, np.invert(np.isnan(act)))]], headers='firstrow')
            )

        # Record time after the action was calculated
        if record_dts:
            t_post_policy = time.time()

        # Ask the environment to perform the simulation step
        obs_next, rew, done, env_info = env.step(act)

        # Record time after the step i.e. the send and receive is completed
        if record_dts:
            t_post_step = time.time()
            dt_policy = t_post_policy - t_start
            dt_step = t_post_step - t_post_policy

        # Record data
        obs_hist.append(obs)
        act_hist.append(act)
        rew_hist.append(rew)
        env_info_hist.append(env_info)
        if record_dts:
            dt_policy_hist.append(dt_policy)
            dt_step_hist.append(dt_step)
            dt_remainder_hist.append(dt_remainder)
        if policy.is_recurrent:
            hidden_hist.append(hidden)
            hidden = hidden_next
        # If an ExplStrat is passed use the policy property, if a Policy is passed use it directly
        if isinstance(getattr(policy, 'policy', policy), (ADNPolicy, NFPolicy)):
            pot_hist.append(getattr(policy, 'policy', policy).potentials.detach().numpy())
            stim_ext_hist.append(getattr(policy, 'policy', policy).stimuli_external.detach().numpy())
            stim_int_hist.append(getattr(policy, 'policy', policy).stimuli_internal.detach().numpy())
        elif isinstance(getattr(policy, 'policy', policy), TwoHeadedPolicy):
            head_2_hist.append(head_2_to)

        # Store the observation for next step (if done, this is the final observation)
        obs = obs_next

        # Render if wanted (actually renders the next state)
        env.render(render_mode, render_step)

        if render_mode.video:
            do_sleep = True
            if pyrado.mujoco_available:
                from pyrado.environments.mujoco.base import MujocoSimEnv
                if isinstance(env, MujocoSimEnv):
                    # MuJoCo environments seem to crash on time.sleep()
                    do_sleep = False
            if do_sleep:
                # Measure time spent and sleep if needed
                t_end = time.time()
                t_sleep = env.dt + t_start - t_end
                if t_sleep > 0:
                    time.sleep(t_sleep)

        # Stochastic reset to make the MDP ergodic (e.g. used for REPS)
        if bernoulli_reset is not None:
            assert 0. <= bernoulli_reset <= 1.
            # Stop the rollout with probability bernoulli_reset (most common choice is 1 - gamma)
            if binomial(1, bernoulli_reset):
                # The complete=True in the returned StepSequence sets the last done element to True
                break

    # --------
    # End loop
    # --------

    if not no_close:
        # Disconnect from EnvReal instance (does nothing for EnvSim instances)
        env.close()

    # Add final observation to observations list
    obs_hist.append(obs)

    # Return result object
    res = StepSequence(
        observations=obs_hist,
        actions=act_hist,
        rewards=rew_hist,
        rollout_info=rollout_info,
        env_infos=env_info_hist,
        complete=True  # the rollout function always returns complete paths
    )

    # Add special entries to the resulting rollout
    if policy.is_recurrent:
        res.add_data('hidden_states', hidden_hist)
    if isinstance(getattr(policy, 'policy', policy), (ADNPolicy, NFPolicy)):
        res.add_data('potentials', pot_hist)
        res.add_data('stimuli_external', stim_ext_hist)
        res.add_data('stimuli_internal', stim_int_hist)
    elif isinstance(getattr(policy, 'policy', policy), TwoHeadedPolicy):
        res.add_data('head_2', head_2_hist)
    if record_dts:
        res.add_data('dts_policy', dt_policy_hist)
        res.add_data('dts_step', dt_step_hist)
        res.add_data('dts_remainder', dt_remainder_hist)

    return res


def after_rollout_query(env: Env, policy: Policy, rollout: StepSequence) -> tuple:
    """
    Ask the user what to do after a rollout has been animated.

    :param env: environment used for the rollout
    :param policy: policy used for the rollout
    :param rollout: collected data from the rollout
    :return: done flag, initial state, and domain parameters
    """
    # Fist entry contains hotkey, second the info text
    options = [
        ['C', 'continue simulation (with domain randomization)'],
        ['N', 'set domain parameters to nominal values and continue'],
        ['F', 'fix the initial state'],
        ['S', 'set a domain parameter explicitly'],
        ['P', 'plot all observations, actions, and rewards'],
        ['PA', 'plot actions'],
        ['PR', 'plot rewards'],
        ['PO', 'plot all observations'],
        ['PO idcs', 'plot selected observations (integers separated by whitespaces)'],
        ['PF', 'plot features (for linear policy)'],
        ['PP', 'plot policy parameters (not suggested for many parameters)'],
        ['PDT', 'plot time deltas (profiling of a real system)'],
        ['PPOT', 'plot potentials, stimuli, and actions'],
        ['E', 'exit']
    ]

    # Ask for user input
    ans = input(tabulate(options, tablefmt='simple') + '\n').lower()

    if ans == 'c' or ans == '':
        # We don't have to do anything here since the env will be reset at the beginning of the next rollout
        return False, None, None

    elif ans == 'n':
        # Get nominal domain parameters
        if isinstance(inner_env(env), SimEnv):
            dp_nom = inner_env(env).get_nominal_domain_param()
            if typed_env(env, ActDelayWrapper) is not None:
                # There is an ActDelayWrapper in the env chain
                dp_nom['act_delay'] = 0
        else:
            dp_nom = None
        return False, None, dp_nom

    elif ans == 'p':
        plot_observations_actions_rewards(rollout)
        return after_rollout_query(env, policy, rollout)

    elif ans == 'pa':
        plot_actions(rollout, env)
        return after_rollout_query(env, policy, rollout)

    elif ans == 'po':
        plot_observations(rollout)
        return after_rollout_query(env, policy, rollout)

    elif 'po' in ans and any(char.isdigit() for char in ans):
        idcs = [int(s) for s in ans.split() if s.isdigit()]
        plot_observations(rollout, idcs_sel=idcs)
        return after_rollout_query(env, policy, rollout)

    elif ans == 'pf':
        plot_features(rollout, policy)
        return after_rollout_query(env, policy, rollout)

    elif ans == 'pp':
        from matplotlib import pyplot as plt
        render_policy_params(policy, env.spec, annotate=False)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == 'pr':
        plot_rewards(rollout)
        return after_rollout_query(env, policy, rollout)

    elif ans == 'pdt':
        plot_dts(rollout.dts_policy, rollout.dts_step, rollout.dts_remainder)
        return after_rollout_query(env, policy, rollout),

    elif ans == 'ppot':
        plot_potentials(rollout)
        return after_rollout_query(env, policy, rollout)

    elif ans == 's':
        if isinstance(env, SimEnv):
            dp = env.get_nominal_domain_param()
            for k, v in dp.items():
                dp[k] = [v]  # cast float to list of one element to make it iterable for tabulate
            print('These are the nominal domain parameters:')
            print(tabulate(dp, headers="keys", tablefmt='simple'))

        # Get the user input
        strs = input('Enter one new domain parameter\n(format: key whitespace value):\n')
        try:
            param = dict(str.split() for str in strs.splitlines())
            # Cast the values of the param dict from str to float
            for k, v in param.items():
                param[k] = float(v)
            return False, None, param
        except (ValueError, KeyError):
            print_cbt(f'Could not parse {strs} into a dict.', 'r')
            after_rollout_query(env, policy, rollout)

    elif ans == 'f':
        try:
            if isinstance(inner_env(env), RealEnv):
                raise pyrado.TypeErr(given=inner_env(env), expected_type=SimEnv)
            elif isinstance(inner_env(env), SimEnv):
                # Get the user input
                str = input(f'Enter the {env.obs_space.flat_dim}-dim initial state'
                            f'(format: each dim separated by a whitespace):\n')
                state = list(map(float, str.split()))
                if isinstance(state, list):
                    state = np.array(state)
                    if state.shape != env.obs_space.shape:
                        raise pyrado.ShapeErr(given=state, expected_match=env.obs_space)
                else:
                    raise pyrado.TypeErr(given=state, expected_type=list)
                return False, state, {}
        except (pyrado.TypeErr, pyrado.ShapeErr):
            return after_rollout_query(env, policy, rollout)

    elif ans == 'e':
        env.close()
        return True, None, {}  # breaks the outer while loop

    else:
        return after_rollout_query(env, policy, rollout)  # recursion
