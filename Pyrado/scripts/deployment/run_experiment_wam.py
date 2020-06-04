""" Execute a trajectory on the real WAM

Dependencies:
    https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0
    https://git.ias.informatik.tu-darmstadt.de/ploeger/robcom_robots

Additional reading:
    Direct control with plain robcom:
    https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0/-/blob/master/examples/robcom/direct_control_example.py
    Ball-in-a-cup demo (using only desired joint position info):
    https://git.ias.informatik.tu-darmstadt.de/klink/ball-in-a-cup-demo/-/blob/master/bic-new.py
    Calibration routine for WAM with 4-dof:
    https://git.ias.informatik.tu-darmstadt.de/ploeger/robcom_robots/-/blob/master/robcom_robots/calibration.py
"""

import os.path as osp
import numpy as np
import robcom_python as r
from robcom_robots.robots import RobcomRobot

from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser


class WAMRobcom(RobcomRobot):

    gain_fade_duration = 500.  # 500 time steps = 1 sec @ f_div = 1.
    p_gains = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
    d_gains = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])
    home_pos = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])
    f_ctrl = 500.  # the frequency of SL's internal PD controller
    n_dof = 7
    lab_ip = '192.168.2.2'

    def __init__(self, **kwargs):
        RobcomRobot.__init__(self, **kwargs)

        # robcom connection
        self.group_name = "RIGHT_ARM"
        self.group = self.robot.get_group([self.group_name])

        # get current state
        self._get_initial_state_and_gains()

        # direct control client
        self._direct_control_client = self.rc_client.create(r.ClosedLoopDirectControl, self.group_name, "")

        # data recording
        self._recorded_trajectory = []
        self._task = None


def main_goto_controlled(des_qpos):
    """
    Uses the clients `goto` command to define desired joint positions over time
    Reference:
        https://git.ias.informatik.tu-darmstadt.de/klink/ball-in-a-cup-demo/-/blob/master/bic-new.py
    """

    start_pos = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])  # starting position
    dt = 0.002  # step size

    # Connect to client
    c = r.Client()
    c.start("127.0.0.1", 2013)  # ip adress and port
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


def main_pd_controlled(des_qpos, des_qvel):
    """
    Use the mujoco_robots class to do gym-like execution of the desired trajectory
    Reference:
        https://git.ias.informatik.tu-darmstadt.de/ploeger/robcom_robots/-/blob/master/robcom_robots/calibration.py
    """
    # Initialize
    wam = WAMRobcom()

    # Reset
    wam.start()
    wam.go_home()
    wam.wait_for_task()

    # Rollout
    for i in range(des_qpos.shape[0]):
        wam.step(des_qpos[i], des_qvel[i])
    wam.wait_for_task()

    # Close
    wam.stop()


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Get desired positions and velocities
    des_qpos = np.load(osp.join(ex_dir, 'des_qpos.npy'))
    des_qvel = np.load(osp.join(ex_dir, 'des_qvel.npy'))

    main_pd_controlled(des_qpos, des_qvel)
    # main_goto_controlled(des_qpos)