"""
Script to determine the voltage offsets on the Qunaser Ball-Balancer device
.. note::
    Make sure that the cylinders are tightly connected to the plate.
    If the state is immediately out of bounds, stop the Simulink model, disconnect and connect again.
"""
import numpy as np
import os.path as osp

import pyrado
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal, QCartPoleStabReal, QCartPoleSwingUpReal
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


def estimate_one_thold_qbb(servo: str, dir_multiplier: int, dact: float, thold_theta: float):
    """
    Apply voltage on one servo in one direction until the servo starts moving.

    :param servo: selected servo motor
    :param dir_multiplier: direction to move in, e.g. +1 for positive voltage i.e. positive angle
    :param dact: delta in actions
    :param thold_theta: threshold for the motor shaft angle measured by the encoders
    :return thold_V: estimated voltage threshold for the selected servo and direction
    """
    # Which servo?
    if servo == 'x':
        theta_idx = 0  # index for theta_x in the observation vector
        act_step = np.array([dact, 0.])
    elif servo == 'y':
        theta_idx = 1  # index for theta_y in the observation vector
        act_step = np.array([0., dact])
    else:
        raise ValueError("`servo` must be either 'x' or 'y'!")

    # Which direction?
    if dir_multiplier == 1 or dir_multiplier == -1:
        act_step *= dir_multiplier
    else:
        raise ValueError(f'`dir_multiplier` must be either 1 or -1, but received {dir_multiplier}!')

    # Initialize
    done = False
    thold_V = np.NAN
    act = np.zeros(2)
    obs_init = env.reset()

    print_cbt(f'Running threshold estimation on servo {servo} in direction {dir_multiplier} ...', 'c', bright=True)
    while not done:
        # Apply voltage and wait
        obs, _, _, _ = env.step(act)

        theta_diff = obs[theta_idx] - obs_init[theta_idx]
        print('act: {}  |  theta_{}: {}  |  theta_{}_diff [deg]: {:.4f}'.format(
            act, servo, obs[theta_idx], servo, theta_diff*180./np.pi))

        if abs(theta_diff) > thold_theta:
            # The servo moved more than the threshold
            thold_V = act[theta_idx]
            done = True
            print_cbt(f'Voltage offset for theta_{servo} in direction {dir_multiplier} is {thold_V}.', 'c')
        else:
            # Prepare for next step
            act += act_step  # increase voltage magnitude

        # Stop if out of bounds
        if not env.obs_space.contains(obs, verbose=True):
            break

    # Send zero action at last
    env.step(np.zeros_like(act))
    return thold_V


def estimate_one_thold_qcp(dir_multiplier: int, dact: float, thold_x: float):
    """
    Apply voltage to the servo in one direction until the servo starts moving.

    :param dir_multiplier: direction to move in, e.g. +1 for positive voltage i.e. positive angle
    :param dact: delta in actions
    :param thold_x: threshold for the motor shaft angle measured by the encoders
    :return thold_V: estimated voltage threshold for the selected servo and direction
    """
    act_step = np.array([dact])
    # Which direction?
    if dir_multiplier == 1 or dir_multiplier == -1:
        act_step *= dir_multiplier
    else:
        raise ValueError(f'`dir_multiplier` must be either 1 or -1, but received {dir_multiplier}!')

    # Initialize
    done = False
    thold_V = np.NAN
    act = np.zeros(1)
    obs_init = env.reset()

    print_cbt(f'Running threshold estimation in direction {dir_multiplier} ...', 'c', bright=True)
    while not done:
        # Apply voltage and wait
        obs, _, _, _ = env.step(act)

        x_diff = obs[0] - obs_init[0]
        print(f'act: {act}  |  x [m]: {obs[0]}  |  x_diff [m]: {x_diff:.5f}')

        if abs(x_diff) > thold_x:
            # The servo moved more than the threshold
            thold_V = act[0]
            done = True
            print_cbt(f'Voltage offset for x in direction {dir_multiplier} is {thold_V}.', 'c')
        else:
            # Prepare for next step
            act += act_step  # increase voltage magnitude

        # Stop if out of bounds
        if not env.obs_space.contains(obs, verbose=True):
            break

    # Send zero action at last
    env.step(np.zeros_like(act))
    return thold_V


if __name__ == '__main__':
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument('--dact', type=float, default=2e-5, help='Step size for the action')
    parser.add_argument('--thold_theta_deg', type=float, default=0.1,
                        help='Threshold for the servo shafts to rotate in degree')
    parser.add_argument('--thold_x', type=float, default=5e-4,
                        help='Threshold for the cart to move in meter')
    args = parser.parse_args()

    if args.env_name == QBallBalancerReal.name:
        # Create the interface for communicating with the device
        env = QBallBalancerReal(args.dt, args.max_steps)
        print_cbt('Set up the QBallBalancerReal environment.', 'c')

        # Initialize
        thold_theta = args.thold_theta_deg/180.*np.pi
        tholds_V = np.zeros((args.num_runs, 2, 2))  # num_runs matrices in format [[-x, +x], [-y, +y]]

        env.reset()  # sends a zero action until sth else is commanded
        input('Servo voltages set to [0, 0]. Prepare and then press enter to start.')

        # Run
        for r in range(args.num_runs):
            tholds_V[r, 0, 0] = estimate_one_thold_qbb('x', -1, args.dact, thold_theta)
            tholds_V[r, 0, 1] = estimate_one_thold_qbb('x', 1, args.dact, thold_theta)
            tholds_V[r, 1, 0] = estimate_one_thold_qbb('y', -1, args.dact, thold_theta)
            tholds_V[r, 1, 1] = estimate_one_thold_qbb('y', 1, args.dact, thold_theta)
        tholds_V_mean = np.mean(tholds_V, axis=0)

        print_cbt(f"Threshold values averages over {args.num_runs} (servos in rows, lower/upper thold in columns):\n"
                  f"{tholds_V_mean}", 'c')

        np.save(osp.join(pyrado.EVAL_DIR, 'volt_thold_qbb', 'qbb_tholds_V.npy'), tholds_V_mean)

    elif args.env_name in [QCartPoleStabReal.name, QCartPoleSwingUpReal.name]:
        # Create the interface for communicating with the device
        env = QCartPoleReal(args.dt, args.max_steps)
        print_cbt('Set up the QCartPoleReal environment.', 'c')

        # Initialize
        tholds_V = np.zeros((args.num_runs, 2))  # num_runs matrices in format [-x, +x]
        env.reset()  # sends a zero action until sth else is commanded

        # Run
        for r in range(args.num_runs):
            tholds_V[r, 0] = estimate_one_thold_qcp(-1, args.dact, args.thold_x)
            tholds_V[r, 1] = estimate_one_thold_qcp(1, args.dact, args.thold_x)
        tholds_V_mean = np.mean(tholds_V, axis=0)

        print_cbt(f"Threshold values averages over {args.num_runs} (servos in rows, lower/upper thold in columns):\n"
                  f"{tholds_V_mean}", 'c')

        np.save(osp.join(pyrado.EVAL_DIR, 'volt_thold_qcp', 'qcp_tholds_V.npy'), tholds_V_mean)

    else:
        raise pyrado.ValueErr(given=args.env_name, eq_constraint=f'{QBallBalancerReal.name}, {QCartPoleStabReal.name},'
                                                                 f' or {QCartPoleSwingUpReal.name}')
