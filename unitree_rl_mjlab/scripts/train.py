"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    # print("task_id:", task_id, ", \n env_cfg:", load_env_cfg(task_id), ", \n agent_cfg:", load_rl_cfg(task_id))
    # task_id: Unitree-G1-Flat , env_cfg: ManagerBasedRlEnvCfg(decimation=4, scene=SceneCfg(num_envs=1, env_spacing=2.0, terrain=TerrainEntityCfg(init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*': 0.0}, joint_vel={'.*': 0.0}), spec_fn=<function EntityCfg.<lambda>.<locals>.<lambda> at 0x7f93562fa7a0>, articulation=None, sort_actuators=False, lights=(LightCfg(name='sun', body='world', mode='fixed', target=None, type='directional', castshadow=True, pos=(0.0, 0.0, 1.5), dir=(0.0, 0.0, -1.0), cutoff=45.0, exponent=10.0),), cameras=(), textures=(TextureCfg(name='groundplane', type='2d', builtin='checker', rgb1=(0.2, 0.3, 0.4), rgb2=(0.1, 0.2, 0.3), width=300, height=300, mark='edge', markrgb=(0.8, 0.8, 0.8)),), materials=(MaterialCfg(name='groundplane', rgba=(1.0, 1.0, 1.0, 1.0), texuniform=True, texrepeat=(4.0, 4.0), reflectance=0.2, texture='groundplane', geom_names_expr=('terrain$',)),), collisions=(), terrain_type='plane', terrain_generator=None, env_spacing=2.0, max_init_terrain_level=5, num_envs=1), entities={'robot': EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0.8), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joint_pos={'.*_hip_pitch_joint': -0.1, '.*_knee_joint': 0.3, '.*_ankle_pitch_joint': -0.2, '.*_shoulder_pitch_joint': 0.35, '.*_elbow_joint': 0.87, 'left_shoulder_roll_joint': 0.18, 'right_shoulder_roll_joint': -0.18}, joint_vel={'.*': 0.0}), spec_fn=<function get_spec at 0x7f9356497ba0>, articulation=EntityArticulationInfoCfg(actuators=(BuiltinPositionActuatorCfg(target_names_expr=('.*_elbow_joint', '.*_shoulder_pitch_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_wrist_roll_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.003609725, frictionloss=0.0, stiffness=14.25062309787429, damping=0.907222843292423, effort_limit=25.0), BuiltinPositionActuatorCfg(target_names_expr=('.*_hip_pitch_joint', '.*_hip_yaw_joint', 'waist_yaw_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.01017752004132231, frictionloss=0.0, stiffness=40.17923863450712, damping=2.557889775413375, effort_limit=88.0), BuiltinPositionActuatorCfg(target_names_expr=('.*_hip_roll_joint', '.*_knee_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.025101924999999997, frictionloss=0.0, stiffness=99.09842777666111, damping=6.308801853496639, effort_limit=139.0), BuiltinPositionActuatorCfg(target_names_expr=('.*_wrist_pitch_joint', '.*_wrist_yaw_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.00425, frictionloss=0.0, stiffness=16.77832748089279, damping=1.06814150219, effort_limit=5.0), BuiltinPositionActuatorCfg(target_names_expr=('waist_pitch_joint', 'waist_roll_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.00721945, frictionloss=0.0, stiffness=28.50124619574858, damping=1.814445686584846, effort_limit=50.0), BuiltinPositionActuatorCfg(target_names_expr=('.*_ankle_pitch_joint', '.*_ankle_roll_joint'), transmission_type=<TransmissionType.JOINT: 'joint'>, armature=0.00721945, frictionloss=0.0, stiffness=28.50124619574858, damping=1.814445686584846, effort_limit=50.0)), soft_joint_pos_limit_factor=0.9), sort_actuators=False, lights=(), cameras=(), textures=(), materials=(), collisions=(CollisionCfg(geom_names_expr=('.*_collision',), contype=1, conaffinity=1, condim={'^(left|right)_foot[1-7]_collision$': 3, '.*_collision': 1}, priority={'^(left|right)_foot[1-7]_collision$': 1}, friction={'^(left|right)_foot[1-7]_collision$': (0.6,)}, solref=None, solimp=None, disable_other_geoms=True),))}, sensors=(ContactSensorCfg(name='feet_ground_contact', primary=ContactMatch(mode='subtree', pattern='^(left_ankle_roll_link|right_ankle_roll_link)$', entity='robot', exclude=()), secondary=ContactMatch(mode='body', pattern='terrain', entity=None, exclude=()), fields=('found', 'force'), reduce='netforce', num_slots=1, secondary_policy='first', track_air_time=True, global_frame=False, history_length=0, debug=False), ContactSensorCfg(name='self_collision', primary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot', exclude=()), secondary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot', exclude=()), fields=('found', 'force'), reduce='none', num_slots=1, secondary_policy='first', track_air_time=False, global_frame=False, history_length=4, debug=False)), extent=2.0, spec_fn=None), observations={'actor': ObservationGroupCfg(terms={'base_ang_vel': ObservationTermCfg(func=<function builtin_sensor at 0x7f9367dfade0>, params={'sensor_name': 'robot/imu_ang_vel'}, noise=UniformNoiseCfg(operation='add', n_min=-0.2, n_max=0.2), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'projected_gravity': ObservationTermCfg(func=<function projected_gravity at 0x7f9367dfaac0>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-0.05, n_max=0.05), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'command': ObservationTermCfg(func=<function generated_commands at 0x7f9367dfad40>, params={'command_name': 'twist'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'phase': ObservationTermCfg(func=<function phase at 0x7f93564af9c0>, params={'period': 0.6, 'command_name': 'twist'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'joint_pos': ObservationTermCfg(func=<function joint_pos_rel at 0x7f9367dfab60>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-0.01, n_max=0.01), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'joint_vel': ObservationTermCfg(func=<function joint_vel_rel at 0x7f9367dfac00>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-1.5, n_max=1.5), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'actions': ObservationTermCfg(func=<function last_action at 0x7f9367dfaca0>, params={}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True)}, concatenate_terms=True, concatenate_dim=-1, enable_corruption=True, history_length=1, flatten_history_dim=True, nan_policy='disabled', nan_check_per_term=True), 'critic': ObservationGroupCfg(terms={'base_ang_vel': ObservationTermCfg(func=<function builtin_sensor at 0x7f9367dfade0>, params={'sensor_name': 'robot/imu_ang_vel'}, noise=UniformNoiseCfg(operation='add', n_min=-0.2, n_max=0.2), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'projected_gravity': ObservationTermCfg(func=<function projected_gravity at 0x7f9367dfaac0>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-0.05, n_max=0.05), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'command': ObservationTermCfg(func=<function generated_commands at 0x7f9367dfad40>, params={'command_name': 'twist'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'phase': ObservationTermCfg(func=<function phase at 0x7f93564af9c0>, params={'period': 0.6, 'command_name': 'twist'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'joint_pos': ObservationTermCfg(func=<function joint_pos_rel at 0x7f9367dfab60>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-0.01, n_max=0.01), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'joint_vel': ObservationTermCfg(func=<function joint_vel_rel at 0x7f9367dfac00>, params={}, noise=UniformNoiseCfg(operation='add', n_min=-1.5, n_max=1.5), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'actions': ObservationTermCfg(func=<function last_action at 0x7f9367dfaca0>, params={}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'base_lin_vel': ObservationTermCfg(func=<function builtin_sensor at 0x7f9367dfade0>, params={'sensor_name': 'robot/imu_lin_vel'}, noise=UniformNoiseCfg(operation='add', n_min=-0.5, n_max=0.5), clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'foot_height': ObservationTermCfg(func=<function foot_height at 0x7f93564af7e0>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=('left_foot', 'right_foot'), site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'foot_air_time': ObservationTermCfg(func=<function foot_air_time at 0x7f93564af060>, params={'sensor_name': 'feet_ground_contact'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'foot_contact': ObservationTermCfg(func=<function foot_contact at 0x7f93564af880>, params={'sensor_name': 'feet_ground_contact'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True), 'foot_contact_forces': ObservationTermCfg(func=<function foot_contact_forces at 0x7f93564af920>, params={'sensor_name': 'feet_ground_contact'}, noise=None, clip=None, scale=None, delay_min_lag=0, delay_max_lag=0, delay_per_env=True, delay_hold_prob=0.0, delay_update_period=0, delay_per_env_phase=True, history_length=0, flatten_history_dim=True)}, concatenate_terms=True, concatenate_dim=-1, enable_corruption=False, history_length=1, flatten_history_dim=True, nan_policy='disabled', nan_check_per_term=True)}, actions={'joint_pos': JointPositionActionCfg(entity_name='robot', clip=None, transmission_type=<TransmissionType.JOINT: 'joint'>, actuator_names=('.*',), scale={'.*_elbow_joint': 0.43857731392336724, '.*_shoulder_pitch_joint': 0.43857731392336724, '.*_shoulder_roll_joint': 0.43857731392336724, '.*_shoulder_yaw_joint': 0.43857731392336724, '.*_wrist_roll_joint': 0.43857731392336724, '.*_hip_pitch_joint': 0.5475464629911068, '.*_hip_yaw_joint': 0.5475464629911068, 'waist_yaw_joint': 0.5475464629911068, '.*_hip_roll_joint': 0.35066146637882434, '.*_knee_joint': 0.35066146637882434, '.*_wrist_pitch_joint': 0.07450087032950714, '.*_wrist_yaw_joint': 0.07450087032950714, 'waist_pitch_joint': 0.43857731392336724, 'waist_roll_joint': 0.43857731392336724, '.*_ankle_pitch_joint': 0.43857731392336724, '.*_ankle_roll_joint': 0.43857731392336724}, offset=0.0, preserve_order=False, use_default_offset=True)}, events={'reset_base': EventTermCfg(func=<function reset_root_state_uniform at 0x7f9367df9bc0>, params={'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (0.0, 0.0), 'yaw': (-3.14, 3.14)}, 'velocity_range': {}}, mode='reset', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0), 'reset_robot_joints': EventTermCfg(func=<function reset_joints_by_offset at 0x7f9367df9d00>, params={'position_range': (-0.0, 0.0), 'velocity_range': (-0.0, 0.0), 'asset_cfg': SceneEntityCfg(name='robot', joint_names=('.*',), joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, mode='reset', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0), 'push_robot': EventTermCfg(func=<function push_by_setting_velocity at 0x7f9367df9da0>, params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.4, 0.4), 'roll': (-0.52, 0.52), 'pitch': (-0.52, 0.52), 'yaw': (-0.78, 0.78)}}, mode='interval', interval_range_s=(5.0, 6.0), is_global_time=False, min_step_count_between_reset=0), 'foot_friction': EventTermCfg(func=<function geom_friction at 0x7f9367dc7ba0>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=('left_foot1_collision', 'left_foot2_collision', 'left_foot3_collision', 'left_foot4_collision', 'left_foot5_collision', 'left_foot6_collision', 'left_foot7_collision', 'right_foot1_collision', 'right_foot2_collision', 'right_foot3_collision', 'right_foot4_collision', 'right_foot5_collision', 'right_foot6_collision', 'right_foot7_collision'), geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False), 'operation': 'abs', 'ranges': (0.3, 1.6), 'shared_random': True}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0), 'encoder_bias': EventTermCfg(func=<function encoder_bias at 0x7f9367dd4540>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False), 'bias_range': (-0.015, 0.015)}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0), 'base_com': EventTermCfg(func=<function body_com_offset at 0x7f9367dd4220>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=('torso_link',), body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False), 'operation': 'add', 'ranges': {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}}, mode='startup', interval_range_s=None, is_global_time=False, min_step_count_between_reset=0)}, seed=None, sim=SimulationCfg(nconmax=None, njmax=300, ls_parallel=True, contact_sensor_maxmatch=64, mujoco=MujocoCfg(timestep=0.005, integrator='implicitfast', impratio=1.0, cone='pyramidal', jacobian='auto', solver='newton', iterations=10, tolerance=1e-08, ls_iterations=20, ls_tolerance=0.01, ccd_iterations=50, gravity=(0.0, 0.0, -9.81), multiccd=False), nan_guard=NanGuardCfg(enabled=False, buffer_size=100, output_dir='/tmp/mjlab/nan_dumps', max_envs_to_dump=5)), viewer=ViewerConfig(lookat=(0.0, 0.0, 0.0), distance=3.0, fovy=None, elevation=-5.0, azimuth=90.0, origin_type=<OriginType.ASSET_BODY: 4>, entity_name='robot', body_name='torso_link', env_idx=0, max_extra_envs=2, enable_reflections=True, enable_shadows=True, height=240, width=320), episode_length_s=20.0, rewards={'track_linear_velocity': RewardTermCfg(func=<function track_linear_velocity at 0x7f93564aff60>, params={'command_name': 'twist', 'std': 0.5}, weight=1.0), 'track_angular_velocity': RewardTermCfg(func=<function track_angular_velocity at 0x7f93562f8040>, params={'command_name': 'twist', 'std': 0.7071067811865476}, weight=1.0), 'body_orientation_l2': RewardTermCfg(func=<function body_orientation_l2 at 0x7f93562f80e0>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=('torso_link',), body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, weight=-1.0), 'pose': RewardTermCfg(func=<class 'src.tasks.velocity.mdp.rewards.variable_posture'>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names='.*', joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False), 'command_name': 'twist', 'std_standing': {'.*': 0.05}, 'std_walking': {'.*hip_pitch.*': 0.5, '.*hip_roll.*': 0.15, '.*hip_yaw.*': 0.15, '.*knee.*': 0.5, '.*ankle_pitch.*': 0.15, '.*ankle_roll.*': 0.1, '.*waist_yaw.*': 0.15, '.*waist_roll.*': 0.1, '.*waist_pitch.*': 0.1, '.*shoulder_pitch.*': 0.15, '.*shoulder_roll.*': 0.1, '.*shoulder_yaw.*': 0.1, '.*elbow.*': 0.1, '.*wrist.*': 0.1}, 'std_running': {'.*hip_pitch.*': 0.5, '.*hip_roll.*': 0.25, '.*hip_yaw.*': 0.25, '.*knee.*': 0.5, '.*ankle_pitch.*': 0.25, '.*ankle_roll.*': 0.1, '.*waist_yaw.*': 0.25, '.*waist_roll.*': 0.1, '.*waist_pitch.*': 0.1, '.*shoulder_pitch.*': 0.25, '.*shoulder_roll.*': 0.1, '.*shoulder_yaw.*': 0.1, '.*elbow.*': 0.1, '.*wrist.*': 0.1}, 'walking_threshold': 0.1, 'running_threshold': 1.5}, weight=1.0), 'body_ang_vel': RewardTermCfg(func=<function body_angular_velocity_penalty at 0x7f93562f8220>, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=('torso_link',), body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, weight=-0.05), 'angular_momentum': RewardTermCfg(func=<function angular_momentum_penalty at 0x7f93562f82c0>, params={'sensor_name': 'robot/root_angmom'}, weight=-0.025), 'is_terminated': RewardTermCfg(func=<function is_terminated at 0x7f9367dfb100>, params={}, weight=-200.0), 'joint_acc_l2': RewardTermCfg(func=<function joint_acc_l2 at 0x7f9367dfb2e0>, params={}, weight=-2.5e-07), 'joint_pos_limits': RewardTermCfg(func=<function joint_pos_limits at 0x7f9367dfb4c0>, params={}, weight=-10.0), 'action_rate_l2': RewardTermCfg(func=<function action_rate_l2 at 0x7f9367dfb380>, params={}, weight=-0.05), 'foot_gait': RewardTermCfg(func=<function feet_gait at 0x7f93562f84a0>, params={'period': 0.6, 'offset': [0.0, 0.5], 'threshold': 0.56, 'command_threshold': 0.1, 'command_name': 'twist', 'sensor_name': 'feet_ground_contact'}, weight=0.5), 'foot_clearance': RewardTermCfg(func=<function feet_clearance at 0x7f93562f8400>, params={'target_height': 0.1, 'command_name': 'twist', 'command_threshold': 0.1, 'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=('left_foot', 'right_foot'), site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, weight=-1.0), 'foot_slip': RewardTermCfg(func=<function feet_slip at 0x7f93562f8540>, params={'sensor_name': 'feet_ground_contact', 'command_name': 'twist', 'command_threshold': 0.1, 'asset_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=('left_foot', 'right_foot'), site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, weight=-0.25), 'soft_landing': RewardTermCfg(func=<function soft_landing at 0x7f93562f8720>, params={'sensor_name': 'feet_ground_contact', 'command_name': 'twist', 'command_threshold': 0.1}, weight=-0.001), 'stand_still': RewardTermCfg(func=<function stand_still at 0x7f93562f87c0>, params={'command_name': 'twist', 'command_threshold': 0.1, 'asset_cfg': SceneEntityCfg(name='robot', joint_names='.*', joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=None, site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), preserve_order=False)}, weight=-1.0), 'self_collisions': RewardTermCfg(func=<function self_collision_cost at 0x7f93565f63e0>, params={'sensor_name': 'self_collision', 'force_threshold': 10.0}, weight=-1.0)}, terminations={'time_out': TerminationTermCfg(func=<function time_out at 0x7f9367dfba60>, params={}, time_out=True), 'fell_over': TerminationTermCfg(func=<function bad_orientation at 0x7f9367dfbd80>, params={'limit_angle': 1.2217304763960306}, time_out=False)}, commands={'twist': UniformVelocityCommandCfg(resampling_time_range=(3.0, 8.0), debug_vis=True, entity_name='robot', heading_command=True, heading_control_stiffness=0.5, rel_standing_envs=0.05, rel_heading_envs=1.0, init_velocity_prob=0.0, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-3.141592653589793, 3.141592653589793)), viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15, scale=0.5))}, curriculum={'command_vel': CurriculumTermCfg(func=<function commands_vel at 0x7f93564ae7a0>, params={'command_name': 'twist', 'velocity_stages': [{'step': 0, 'lin_vel_x': (-0.5, 1.0), 'lin_vel_y': (-0.5, 0.5), 'ang_vel_z': (-1.0, 1.0)}, {'step': 120000, 'lin_vel_x': (-1.0, 2.0), 'lin_vel_y': (-1.0, 1.0)}]})}, metrics={'mean_action_acc': MetricsTermCfg(func=<function mean_action_acc at 0x7f9367dfa660>, params={})}, is_finite_horizon=False, scale_rewards_by_dt=True) , agent_cfg: RslRlOnPolicyRunnerCfg(seed=42, num_steps_per_env=24, max_iterations=10001, obs_groups={'actor': ('actor',), 'critic': ('critic',)}, save_interval=100, experiment_name='g1_velocity', run_name='', logger='wandb', wandb_project='mjlab', wandb_tags=(), resume=False, load_run='.*', load_checkpoint='model_.*.pt', clip_actions=None, upload_model=True, class_name='OnPolicyRunner', actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True, cnn_cfg=None, distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}, class_name='MLPModel'), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True, cnn_cfg=None, distribution_cfg=None, class_name='MLPModel'), algorithm=RslRlPpoAlgorithmCfg(num_learning_epochs=5, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.99, lam=0.95, entropy_coef=0.01, desired_kl=0.01, max_grad_norm=1.0, value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, normalize_advantage_per_mini_batch=False, optimizer='adam', share_cnn_encoders=False, class_name='PPO'))
    # agent_cfg: RslRlOnPolicyRunnerCfg(seed=42, num_steps_per_env=24, max_iterations=10001, obs_groups={'actor': ('actor',), 'critic': ('critic',)}, save_interval=100, experiment_name='g1_velocity', run_name='', logger='wandb', wandb_project='mjlab', wandb_tags=(), resume=False, load_run='.*', load_checkpoint='model_.*.pt', clip_actions=None, upload_model=True, class_name='OnPolicyRunner', actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True, cnn_cfg=None, distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}, class_name='MLPModel'), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True, cnn_cfg=None, distribution_cfg=None, class_name='MLPModel'), algorithm=RslRlPpoAlgorithmCfg(num_learning_epochs=5, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.99, lam=0.95, entropy_coef=0.01, desired_kl=0.01, max_grad_norm=1.0, value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, normalize_advantage_per_mini_batch=False, optimizer='adam', share_cnn_encoders=False, class_name='PPO'))
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
