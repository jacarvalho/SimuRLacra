"""
Copied from the Isaac Gym documentation
"""
import random
from carbongym import gymapi


if __name__ == '__main__':
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, 0, gymapi.SIM_FLEX)

    asset_root = "/home/muratore/Software/SimuRLacra/thirdParty/isaac_gym/assets"
    asset_file = "mjcf/humanoid_20_5.xml"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set up the env grid
    num_envs = 64
    envs_per_row = 8
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        height = random.uniform(1.0, 2.5)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, height, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
        actor_handles.append(actor_handle)

    # get current simulation params
    params = gymapi.SimParams()
    gym.get_sim_params(sim, params)

    # set custom params
    # these values are the default and could be skipped during initialization
    # only custom values, different from the default should be manually set
    params.dt = 1/60
    params.substeps = 2
    params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)

    # physics engine (Flex) specific parameters
    params.flex.solver_type = 5
    params.flex.num_outer_iterations = 4
    params.flex.num_inner_iterations = 20
    params.flex.relaxation = 0.8
    params.flex.warm_start = 0.5
    gym.set_sim_params(sim, params)

    # add a viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update viewer and sync with real time
        gym.step_graphics(sim)  # synchronizes the visual representation of the simulation with the physics state
        gym.draw_viewer(viewer, sim, True)  # renders the latest snapshot in the viewer
        gym.sync_frame_time(sim)  # add at the end

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
