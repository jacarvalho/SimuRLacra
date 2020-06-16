""" Execute a trajectory on the real WAM using robcom's GoTo command

Dependencies:
    https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0

Additional reading:
    Ball-in-a-cup demo:
    https://git.ias.informatik.tu-darmstadt.de/klink/ball-in-a-cup-demo/-/blob/master/bic-new.py
"""

import os.path as osp
import numpy as np
import robcom_python as r

from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser


def run_direct_control(des_qpos, des_qvel):

    def callback(jg, eg, data_provider):
        nonlocal n
        nonlocal time_step
        nonlocal qpos
        nonlocal qvel

        if time_step >= n:
            return True

        dpos = des_qpos[time_step].tolist()
        dvel = des_qvel[time_step].tolist()

        pos = np.array(jg.get(r.JointState.POS))
        vel = np.array(jg.get(r.JointState.VEL))
        qpos.append(pos)
        qvel.append(vel)

        jg.set(r.JointDesState.POS, dpos)
        jg.set(r.JointDesState.VEL, dvel)

        time_step += 1

        return False

    # Connect to client
    c = r.Client()
    c.start('192.168.2.2', 2013)  # ip adress and port
    print("Connected to client.")

    # Reset the robot to the initial position
    gt = c.create(r.Goto, "RIGHT_ARM", "")
    gt.add_step(5.0, start_pos)
    print("Moving to initial position")
    gt.start()
    gt.wait_for_completion()
    print("Reached initial position")

    # Read out some states
    group = c.robot.get_group(["RIGHT_ARM"])
    home_pos = np.array(group.get(r.JointState.POS))
    p_gains = np.array(group.get(r.JointState.P_GAIN))
    d_gains = np.array(group.get(r.JointState.D_GAIN))
    print("Initial POS:", home_pos)
    print("P Gain:", p_gains)
    print("D Gain:", d_gains)

    # Global callback attributes
    n = des_qpos.shape[0]
    time_step = 0
    qpos = []
    qvel = []

    # Start the direct control
    dc = c.create(r.ClosedLoopDirectControl, "RIGHT_ARM", "")
    print("Executing trajectory")
    dc.start(False, 1, callback, ['POS', 'VEL'], [], [])
    dc.wait_for_completion()
    print("Finished execution.")

    print('Measured positions:', np.array(qpos).shape)
    print('Measured velocities:', np.array(qvel).shape)

    c.stop()
    print('Connection closed.')


def run_goto(des_qpos, start_pos, dt):
    # Connect to client
    c = r.Client()
    c.start('192.168.2.2', 2013)  # ip adress and port
    print("Connected to client.")

    # Reset the robot to the initial position
    gt = c.create(r.Goto, "RIGHT_ARM", "")
    gt.add_step(5.0, start_pos)
    print("Moving to initial position")
    gt.start()
    gt.wait_for_completion()
    print("Reached initial position")

    gt = c.create(r.Goto, "RIGHT_ARM", "")
    for i in range(0, des_qpos.shape[0]):
        gt.add_step(dt, des_qpos[i, :])
    print("Executing trajectory")
    gt.start()
    gt.wait_for_completion()
    print("Finished execution.")

    c.stop()
    print('Connection closed.')


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Get desired positions and velocities
    des_qpos = np.load(osp.join(ex_dir, 'des_qpos.npy'))
    des_qvel = np.load(osp.join(ex_dir, 'des_qvel.npy'))

    start_pos = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])  # starting position
    dt = 0.002  # step size

    run_goto(des_qpos, start_pos, dt)
    # run_direct_control(des_qpos, des_qvel)
