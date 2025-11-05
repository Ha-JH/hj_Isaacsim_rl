# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from rirolab_assets.robots.shadow_hand_lite import SHADOW_HAND_LITE_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from rirolab_assets import LOCAL_ASSETS_DIR

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # "mass_distribution_params": (0.5, 1.5),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class ShadowHandLiteEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # action_space = 20
    action_space = 13
    observation_space = 157  # (full)  for rsl-rl

    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    actuated_joint_names = [
        "rh_FFJ4",
        "rh_FFJ3",
        "rh_FFJ2",
        "rh_MFJ4",
        "rh_MFJ3",
        "rh_MFJ2",
        "rh_RFJ4",
        "rh_RFJ3",
        "rh_RFJ2",
        "rh_THJ5",
        "rh_THJ4",
        "rh_THJ2",
        "rh_THJ1",
    ]
    fingertip_body_names = [
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_thdistal",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/007_tuna_fish_can.usd",
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/spam_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            # scale=(0.08, 0.08, 0.08) # for Dexcube
            # scale=(0.55, 0.55, 0.55)
            # scale=(0.1, 0.1, 0.1) #for spam
            # scale=(0.08, 0.08, 0.08) #for lemon
            # scale=(0.065, 0.065, 0.065) #for Mug
             scale=(0.65, 0.65, 0.65) #for spam
            # scale=(0.9, 0.9, 0.9) #fishcan
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.19, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.005, -0.23, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/spam_tex.usd",
                # usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/007_tuna_fish_can.usd",

                # scale=(0.55, 0.55, 0.55) 
                # scale=(0.095, 0.095, 0.095) #for peach
                # scale=(0.08, 0.08, 0.08) #for lemon
                # scale=(0.9, 0.9, 0.9) #fishcan
                # scale=(0.065, 0.065, 0.065) #for Mug
                scale=(0.65, 0.65, 0.65) #for spam
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)

    # reset
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 


    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -8.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 4   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0007  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

####################################################################################################3

@configclass
class ShadowHandLiteEnvLemonCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/007_tuna_fish_can.usd",
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/lemon_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            # scale=(0.08, 0.08, 0.08) # for Dexcube
            # scale=(0.55, 0.55, 0.55)
            # scale=(0.095, 0.095, 0.095) #for peach
            scale=(0.08, 0.08, 0.08) #for lemon

            # scale=(1.0, 1.0, 1.0)
            # scale=(0.9, 0.9, 0.9) #fishcan
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.19, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.005, -0.23, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/teddy_bear.usd",
                # usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/007_tuna_fish_can.usd",
                # scale=(0.55, 0.55, 0.55) 
                # scale=(0.095, 0.095, 0.095) #for peach
                # scale=(0.08, 0.08, 0.08) #for lemon

                scale=(0.004, 0.004, 0.004) #for teddybear
                # scale=(0.9, 0.9, 0.9) #fishcan
            )
        },
    )





    
    # reset
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -10.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 2.5   #1.0 -> 2.5
    rot_eps = 0.3 #0.1 -> 0.3

    action_penalty_scale = -0.0007  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -15
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    success_tolerance = 0.18 #10 degrees sodp

    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class ShadowHandLiteEnvMugCfg(ShadowHandLiteEnvCfg): 
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/mug_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.07, 0.07, 0.07) #for Mug

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/mug_tex.usd",
                scale=(0.07, 0.07, 0.07) #for Mug
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 
    # reset_position_noise = 0.01  # range of position at reset

    reset_position_noise = 0.00
    reset_dof_pos_noise = 0.1  # range of dof pos at reset
    reset_dof_vel_noise = 0.05  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -8.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 4   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0012  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 150
    fall_penalty = -20
    fall_dist = 0.24
    vel_obs_scale = 0.2

    # success_tolerance = 0.175 # 10degrees for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    success_tolerance = 0.35 #20 degrees sodp 

    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0



#glue Train and Play configs
@configclass
class ShadowHandLiteEnvGlueCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/glue_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.09, 0.09, 0.09) #for Glue

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/glue_tex.usd",
                scale=(0.09, 0.09, 0.09) #for Glue
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -10.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 5.0   #1.0 -> 2.5
    rot_eps = 0.3 #0.1 -> 0.3

    action_penalty_scale = -0.0007  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -15
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    success_tolerance = 0.05 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

#Pillcase Train and Play configs
@configclass
class ShadowHandLiteEnvPillcaseCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/pillcase_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.07, 0.07, 0.07) #for Pillcase

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/pillcase_tex.usd",
                scale=(0.07, 0.07, 0.07) #for Pillcase
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -8.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 4   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0007  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


#Redblock Train and Play configs
@configclass
class ShadowHandLiteEnvRedblockCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/redblock_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.07, 0.07, 0.07) #for Redblock

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.50), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/redblock_tex.usd",
                scale=(0.07, 0.07, 0.07) #for Redblock
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.1  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 10   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -25
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


#Spam Train and Play configs
@configclass
class ShadowHandLiteEnvSpamCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/spam_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.65, 0.65, 0.65) #for Spam

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/spam_tex.usd",
                scale=(0.65, 0.65, 0.65) #for Spam
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 10.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0



#Bowl Train and Play configs
@configclass
class ShadowHandLiteEnvBowlCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/bowl_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.45, 0.45, 0.45) #for Bowl

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/bowl_tex.usd",
                scale=(0.45, 0.45, 0.45) #for Bowl
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    #손바닥에 갇혀있는경우 

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 20.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -25
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0




#Pitcher Train and Play configs
@configclass
class ShadowHandLiteEnvPitcherCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/glue_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            # scale=(0.35, 0.35, 0.35) #for Pitcher
            scale=(0.1, 0.1, 0.1) #for Glue


        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/glue_tex.usd",
                # scale=(0.35, 0.35, 0.35) #for Pitcher
                # scale=(0.35, 0.35, 0.35) #for Pitcher
                scale=(0.1, 0.1, 0.1) #for Glue
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    #손바닥에 갇혀있는경우 

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 20.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -25
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    success_tolerance = 0.5 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

#Teapot Train and Play configs
@configclass
class ShadowHandLiteEnvTeapotCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/teapot_tex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(0.8, 0.8, 0.8) #for Teapot

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/teapot_tex.usd",
                scale=(0.8, 0.8, 0.8) #for Teapot
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    #손바닥에 갇혀있는경우 

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 10.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -25
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.5 #30 degrees sodp 
    success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 


    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0    


#Bulb Train and Play configs
@configclass
class ShadowHandLiteEnvBulbCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/light_bulb.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            scale=(75.0, 75.0, 75.0) #for Bulb

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.23, 0.55), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/light_bulb.usd",
                scale=(75.0, 75.0, 75.0) #for Bulb
            )
        },
    )
    
    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 

    # reset_position_noise = 0.01  # range of position at reset
    #손바닥에 갇혀있는경우 

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 30.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.00005  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -15
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    # success_tolerance = 0.60 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 

    success_tolerance = 1.0 #100 degrees sodp 

    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0    



# Teddybear(Deformable) Train and Play configs
@configclass
class ShadowHandLiteEnvTeddybearCfg(ShadowHandLiteEnvCfg):
    # in-hand object
    deformable_object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/teddy_bear_ver2.usd",
            usd_path=f"{LOCAL_ASSETS_DIR}/objects/teddy_bear_test.usda",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                # enable_gravity=True,
                # use_mass_scaling=False,
                solver_position_iteration_count=100,
                vertex_velocity_damping=1.0,
                settling_threshold=0.05,
                sleep_damping=10.0,
                self_collision=True,
                simulation_hexahedral_resolution=10,
                collision_simplification=True,
                sleep_threshold=0.005,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            scale=(0.003, 0.003, 0.003) #for Teddy Bear
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=[-0.01, -0.23, 0.55], rot=[1.0, 0.0, 0.0, 0.0]), #wxyz
    )

    decimation = 3
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # physx=PhysxCfg(
        #     bounce_threshold_velocity=0.2,
        #     gpu_max_rigid_contact_count=2**23,
        #     gpu_max_rigid_patch_count=2**23,
        # ),
    )

    # set deformable object properties
    youngs_modulus = 5.0e5   # 500,000 정도. 1e6까지 올려도 됨
    damping = 0.01           # 출렁임 줄이기 위해 0~0.002 사이
    damping_scale = 0.2      # 전체 damping 효과는 약간만
    poisson_ratio = 0.25     # 0.2~0.3 사이면 자연스럽고 과도한 부피보존은 안 함
    dynamic_friction = 0.5   # 이건 그대로 둬도 됨

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{LOCAL_ASSETS_DIR}/objects/Props/teddy_bear.usd",
                scale=(0.003, 0.003, 0.003) #for Teddy Bear
            )
        },
    )
    
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)

    #rt​= wdist​⋅rdist​​​+(회전) 일치 wrot​⋅rrot​​​+행동 L2 패널티 wact​⋅ract​​​+성공 보너스 breach​⋅1{dt​<success_tolerance 
    # reset_position_noise = 0.01  # range of position at reset
    #손바닥에 갇혀있는경우 
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset

    # reward scales
    dist_reward_scale = -1.0   #-10.0 -> -5.0 # distance from palm to object
    rot_reward_scale = 100.0   #1.0 -> 5
    rot_eps = 0.2 #0.1 -> 0.3

    action_penalty_scale = -0.0001  #*0.0002->0.0003 #more movement?

    reach_goal_bonus = 250
    fall_penalty = -100
    fall_dist = 0.24
    vel_obs_scale = 0.2
    # success_tolerance = 0.1 # for dexcube
    success_tolerance = 0.50 #30 degrees sodp 
    # success_tolerance = 0.35 #20 degrees sodp 
    # success_tolerance = 0.18 #10 degrees sodp 

    # max_consecutive_success = 0
    max_consecutive_success = 10
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0    
####################################################################################################3
@configclass
class ShadowHandLiteOpenAIEnvCfg(ShadowHandLiteEnvCfg):
    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 20
    observation_space = 42
    state_space = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
