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


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Get desired positions and velocities
    des_qpos = np.load(osp.join(ex_dir, 'des_qpos.npy'))

    start_pos = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])  # starting position
    dt = 0.002  # step size

    # Connect to client
    ip = input('Enter Client IP address: ')
    c = r.Client()
    c.start(ip, 2013)  # ip adress and port
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