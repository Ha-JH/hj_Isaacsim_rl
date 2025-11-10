# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.managers import RecorderManager

import gymnasium as gym


if TYPE_CHECKING:
    # from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from rirolab_tasks.tasks.direct.shadow_hand_lite.shadow_hand_lite_env_cfg import ShadowHandLiteEnvCfg

class InHandManipulationDeformEnv(DirectRLEnv):
    cfg: ShadowHandLiteEnvCfg

    def __init__(self, cfg: ShadowHandLiteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        ## set deformable object physical parameters here if needed ####################################################################

        env_ids = torch.arange(self.num_envs).reshape(-1, 1) # Shape: [num_envs, 1]
        if hasattr(self.cfg, "youngs_modulus"):
            youngs_modulus = torch.full((self.num_envs, ), self.cfg.youngs_modulus) # Shape: [num_envs, 1]
            self.scene.deformable_objects['deformable_object'].material_physx_view.set_youngs_modulus(youngs_modulus, env_ids)
        if hasattr(self.cfg, "damping"):
            damping = torch.full((self.num_envs, ), self.cfg.damping) # Shape: [num_envs, 1]
            self.scene.deformable_objects['deformable_object'].material_physx_view.set_damping(damping, env_ids)
        if hasattr(self.cfg, "dynamic_friction"):
            dynamic_friction = torch.full((self.num_envs, ), self.cfg.dynamic_friction) # Shape: [num_envs, 1]
            self.scene.deformable_objects['deformable_object'].material_physx_view.set_dynamic_friction(dynamic_friction, env_ids)
        if hasattr(self.cfg, "damping_scale"):
            damping_scale = torch.full((self.num_envs, ), self.cfg.damping_scale) # Shape: [num_envs, 1]
            self.scene.deformable_objects['deformable_object'].material_physx_view.set_damping_scale(damping_scale, env_ids)
        if hasattr(self.cfg, "poisson_ratio"):
            poissons_ratio = torch.full((self.num_envs, ), self.cfg.poisson_ratio) # Shape: [num_envs, 1]
            self.scene.deformable_objects['deformable_object'].material_physx_view.set_poissons_ratio(poissons_ratio, env_ids)


        default_world_pos = self.deform_obj.data.default_nodal_state_w[..., :3].clone() # Shape: [B, N, 3]
        default_centroid = torch.mean(default_world_pos, dim=1, keepdim=True) # Shape: [B, 1, 3] #
        self.default_vertices_local = default_world_pos - default_centroid # Shape: [B, N, 3] # SVD를 위한 local 좌표계로 변환(중심점 기준)
        # self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- SVD 결과를 저장 Tensor 생성 ---
        self.current_object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) # 현재 물체 위치
        self.current_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device) # 현재 물체 회전 (쿼터니언)
        self.current_object_rot[:, 0] = 1.0 # (x,y,z,w) 순서로 w=1 초기화  --> wxyz 로 바꿔줘야함 (1,0,0,0)


        # config의 rot이 [w, x, y, z] 순서라고 가정합니다.
        init_rot_wxyz = self.cfg.deformable_object_cfg.init_state.rot
        self.init_object_rot = torch.tensor(init_rot_wxyz, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

        # --- [수정] SVD 축 뒤집힘 방지를 위한 이전 V_T 저장 버퍼 ---
        self.prev_V_T = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1) # Shape: [B, 3, 3]


        self.in_hand_pos = default_centroid.squeeze(1) - self.scene.env_origins[env_ids].squeeze(1)  # <--(B, 3) 형태로 변경 / SVD의 중심점 (goal point comparison 용)
        self.in_hand_pos[:, 2] -= 0.04 # z 약간 아래로 보정, SVD와 비교하는 용으로 goal point 정의

        ################################################################################################################################
        self.num_hand_dofs = self.hand.num_joints
        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device) #buffer for hand dof targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device) #buffer for previous hand dof targets
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device) #buffer for current hand dof targets

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0 #(1,0,0,0) 초기화 --> wxyz
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        
        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.recorder_manager = None
        if hasattr(self.cfg, "recorders") and self.cfg.recorders is not None:
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        self.success_counter = 0

        ################################################################################################
        


    # def DeformableCalculations(self):
    #     """
    #     Additional calculation for deformable object handling.
    #     """
    #     current_vertices = self.deform_obj.data.point_pos_w # Shape: [B, N, 3]
    #     initial_vertices = self.deform_obj.data.default_point_pos_w # Shape: [B, N, 3]


    #     current_centroid = torch.mean(current_vertices, dim=1, keepdim=True)
    #     current_vertices_local = current_vertices - current_centroid

    #     transform = corresponding_points_alignment(
    #         self.default_vertices_local, 
    #         current_vertices_local, 
    #         allow_scaling=False
    #     )
    #     R_curr = transform.R  # Shape: [B, 3, 3]
    #     return 



    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)

        # self.object = RigidObject(self.cfg.object_cfg) #Rigid object
        self.deform_obj = DeformableObject(self.cfg.deformable_object_cfg) # Deformable object ################################ deform object emdfhr
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand

        # self.scene.rigid_objects["object"] = self.object
        self.scene.deformable_objects["deformable_object"] = self.deform_obj ################################ deform object emdfhr

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist


        #torch.isfinite()는 NaN과 Inf를 모두 감지
        # object_pos의 모든 차원(x,y,z)이 유효한지 확인
        is_pos_valid = torch.all(torch.isfinite(self.object_pos), dim=-1) 
        # object_linvel의 모든 차원(x,y,z)이 유효한지 확인
        is_vel_valid = torch.all(torch.isfinite(self.object_linvel), dim=-1)
        
        simulation_exploded = torch.logical_not(is_pos_valid & is_vel_valid)
        # 물체가 떨어졌거나, 시뮬레이션이 폭발했다면 리셋
        terminations = out_of_reach | simulation_exploded

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0

            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        # object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # change shape [1, max_sim_vertices_per_body, 6] into [num_envs, max_sim_vertices_per_body, 6] by stacking
        # object_default_state = self.deform_obj.data.default_nodal_state_w.clone().unsqueeze(0).repeat(len(env_ids), 1, 1) # Shape: [num_envs, N, 6]
        # object_default_state = self.deform_obj.data.default_nodal_state_w.clone().squeeze() #### deform object [B, N, 6]
        object_default_state = self.deform_obj.data.default_nodal_state_w.clone()[env_ids] #### deform object [B, N, 6]

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) # noise for position
        """
        default_root_state(rigid body)

        Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).       
            The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
            of the center of mass frame.
        """

        """

        default_nodal_state_w(deformable body)
        Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
        Shape is (num_instances, max_sim_vertices_per_body, 6).

        """

        #position noise 적용 + env origin 더해주기
        # object_default_state[..., :3] = (
        #     object_default_state[..., :3] 
        #     + self.cfg.reset_position_noise * pos_noise.unsqueeze(1)  # position noise
        #     + self.scene.env_origins[env_ids].unsqueeze(1)  # global position offset(origin)
        # )


        object_default_state[..., :3] = (
            object_default_state[..., :3] # (max_sim_vertices_per_body, 3)
            + self.cfg.reset_position_noise * pos_noise.unsqueeze(1)  # position noise (len(env_ids), 3)
            # + self.scene.env_origins[env_ids].unsqueeze(1)  # global position offset(origin) (len(env_ids), 3
        ) 
        #속도를 0으로 초기화
        object_default_state[..., 3:] = torch.zeros_like(object_default_state[..., 3:])

        #rotation noise for Rigid body
        #rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        # reset deformable object state
        self.deform_obj.write_nodal_state_to_sim(object_default_state, env_ids)
        
      
      
        """
        self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
        self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)
        """


        # reset hand (그대로 두기)
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0

        # --- [수정] SVD '메모리' 리셋 ---
        # 리셋되는 환경(env_ids)에 한해 prev_V_T 버퍼를 항등 행렬(기본값)로 초기화합니다.
        self.prev_V_T[env_ids] = torch.eye(3, device=self.device)


        self._compute_intermediate_values()

        if self.recorder_manager is not None:
            self.recorder_manager.reset(env_ids)



    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)

        # marker rotation 좌표 고정용
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        # new_rot = 
        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
      
#########################################################################################3
        # 1. 실제 변형체의 현재 정점 정보
        current_vertices = self.deform_obj.data.nodal_pos_w # Shape: [B, N, 3] 
        
        # 2. 현재 중심점 및 중심화된 정점
        current_centroid = torch.mean(current_vertices, dim=1, keepdim=True)
        current_vertices_local = current_vertices - current_centroid

        # # 3. SVD로 현재 회전(R_curr) 계산 (JIT 함수 호출)
        # R_curr = compute_rotation_from_svd(
        #     self.default_vertices_local, 
        #     current_vertices_local
        # ) # Shape: [B, 3, 3]
        
        # 3. SVD로 현재 회전(R_curr) 계산 (JIT 함수 호출)
        R_curr, new_V_T = compute_rotation_from_svd( # [수정] 2개의 값을 반환받음
            self.default_vertices_local, 
            current_vertices_local,
            self.prev_V_T # [수정] 이전 V_T 값을 인자로 전달
        ) # Shape: [B, 3, 3]
        # 3.5. [수정] 다음 스텝을 위해 V_T 상태 저장 (detach로 그래프 분리)
        self.prev_V_T = new_V_T.detach()


        # 4. 회전 행렬을 쿼터니언으로 변환 (JIT 함수 호출)
        q_curr_relative_wxyz = matrix_to_quaternion_wxyz(R_curr) # Shape: [B, 4] (w, x, y, z)

        self.object_rot = quat_mul(q_curr_relative_wxyz, self.init_object_rot) # Shape: [B, 4] (w, x, y, z)  # 초기 회전 보정 적용
        
        # 5. 원본 변수에 SVD 결과 덮어쓰기
        self.object_pos = current_centroid.squeeze(1) - self.scene.env_origins
       
        # 속도 정보는 DeformableObject의 root_vel_w를 그대로 사용
        # ... (속도 계산 부분은 그대로) ...
        current_nodal_vel = self.deform_obj.data.nodal_vel_w  # Shape: [B, N, 3]
        self.object_linvel = torch.mean(current_nodal_vel, dim=1) # Shape: [B, 3]
        self.object_angvel = torch.zeros_like(self.object_linvel) # Shape: [B, 3]
        self.object_velocities = torch.cat([self.object_linvel, self.object_angvel], dim=-1)

#########################################################################################3

        # data for Rigid object(참고용) 
        # self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # self.object_rot = self.object.data.root_quat_w
        # self.object_velocities = self.object.data.root_vel_w
        # self.object_linvel = self.object.data.root_lin_vel_w
        # self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot, # (w, x, y, z) 순서
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)), #object_rot (w, x, y, z) 순서
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot, # (w, x, y, z) 순서
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)


#######################################################################################3
        # code for the evaluation!
        # self.successes cumulated success count

        if len(self.successes) == 1: # skip evaluation if it is training code 

            if self.successes == 0:

                self.success_counter = 0

            if self.successes > self.success_counter:
                self.success_counter += 1
                self.extras["is_success"] = self.success_counter
            else:
                self.extras["is_success"] = False
        else:
            self.extras["is_success"] = None


#######################################################################################3
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()


        if self.recorder_manager is not None and len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self._get_observations() # make sure obs
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            if self.recorder_manager is not None:
                self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            if self.recorder_manager is not None:
                self.recorder_manager.record_post_reset(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras



@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))

    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rotation_from_svd(P_src: torch.Tensor, P_tgt: torch.Tensor, prev_V_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SVD(Kabsch 알고리즘)를 사용하여 P_src를 P_tgt에 정렬하는
    최적의 회전 행렬 R을 계산 (JIT 호환)

    Args:
        P_src: 소스 점 구름 (중심점이 0이어야 함) (Shape: [B, N, 3])
        P_tgt: 타겟 점 구름 (중심점이 0이어야 함) (Shape: [B, N, 3])

    Returns:
        회전 행렬 R (Shape: [B, 3, 3])
    """
    
    # 공분산 행렬 H = P_src_T * P_tgt
    H = torch.bmm(P_src.transpose(1, 2), P_tgt) # Shape: [B, 3, 3]

    # SVD 수행: H = U * S * V_T
    # torch.linalg.svd는 V가 아닌 V_T를 반환합니다.
    U, S, V_T = torch.linalg.svd(H) # V_T Shape: [B, 3, 3]

    # 3. [추가됨] 축 연속성 보장 (Temporal Coherence Check)
    #    새 V_T의 첫 행(첫 번째 주축)과 이전 V_T의 첫 행의 부호가 같은지 확인
    #    (내적을 사용해 부호가 반대인(-1) 환경을 찾음)
    dot_product = torch.sum(V_T[:, 0, :] * prev_V_T[:, 0, :], dim=1) # Shape: [B]
    flip_sign = torch.sign(dot_product) # Shape: [B]
    flip_sign[flip_sign == 0] = 1.0 # 0 방지
    
    # 부호를 뒤집어야 하는(-1) env에 대해 U와 V_T의 부호를 모두 변경
    flip_tensor = flip_sign.unsqueeze(-1).unsqueeze(-1) # Shape: [B, 1, 1]
    V_T = V_T * flip_tensor
    U = U * flip_tensor

################################3
    # 최적의 회전 행렬 R = V * U_T
    # V = V_T.transpose(1, 2)
    R = torch.bmm(V_T.transpose(1, 2), U.transpose(1, 2)) # Shape: [B, 3, 3]

    # R의 행렬식(determinant)이 -1이 되는 경우 (반전된 경우)를 방지
    det_R = torch.linalg.det(R) # Shape: [B]
    
    # det_R < 0 인 envs에 대해 V_T의 마지막 행의 부호를 뒤집음
    V_T_new = V_T.clone()
    V_T_new[:, -1, :] *= torch.sign(det_R).unsqueeze(-1)
    
################################3

    # 보정된 최적 회전 행렬
    R_no_reflect = torch.bmm(V_T_new.transpose(1, 2), U.transpose(1, 2))

    return R_no_reflect, V_T_new  

@torch.jit.script
def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    
    """

    회전 행렬을 (w, x, y, z) 순서의 쿼터니언으로 변환 (JIT 호환)

    """
    B = matrix.shape[0]
    # q = torch.empty((B, 4), dtype=matrix.dtype, device=matrix.device)

    M = matrix.reshape(B, 3, 3)
    trace = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    
    # TF3D/Pytorch3D (JIT 호환)
    m00, m01, m02 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    m10, m11, m12 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    m20, m21, m22 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

    w_sq = 0.25 * (1.0 + trace)
    w = torch.sqrt(torch.clamp(w_sq, min=0.0))

    x_sq = w_sq - 0.5 * (m11 + m22)
    x = torch.sqrt(torch.clamp(x_sq, min=0.0)) * torch.sign(m21 - m12)

    y_sq = w_sq - 0.5 * (m00 + m22)
    y = torch.sqrt(torch.clamp(y_sq, min=0.0)) * torch.sign(m02 - m20)

    z_sq = w_sq - 0.5 * (m00 + m11)
    z = torch.sqrt(torch.clamp(z_sq, min=0.0)) * torch.sign(m10 - m01)

    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
   
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    is_exploded = torch.isnan(goal_dist) | torch.isnan(rot_dist)
    # print(f"rot_dist: {rot_dist*180.0/3.1415926}, object_rot: {object_rot}, target_rot: {target_rot}")

    print(f"object_rot: {object_rot}")


    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # --- [수정 2] 폭발하지 않았을 때만 보너스와 페널티를 계산 ---
    # Find out which envs hit the goal and update successes count
    goal_reached = (torch.abs(rot_dist) <= success_tolerance) & ~is_exploded
    goal_resets = torch.where(goal_reached, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # successes = successes + goal_resets
    




    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    # --- [수정 3: 가장 중요] 폭발했다면, 모든 보상을 무시하고 fall_penalty 값으로 덮어씁니다. ---
    # 이것이 NaN 리워드를 막는 핵심 코드입니다.
    reward = torch.where(is_exploded, torch.full_like(reward, fall_penalty), reward)

    # --- [수정 4] 폭발도 리셋으로 간주하여 cons_successes 계산에 반영 ---
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = resets | is_exploded # 폭발도 리셋에 포함



    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes