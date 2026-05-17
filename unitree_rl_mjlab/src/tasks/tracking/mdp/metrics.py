from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Mean Per-Keybody Position Error (MPKPE).

  计算每个关键身体部位的平均位置误差 (MPKPE)。
  MPKPE 衡量了在世界坐标系下，所有关键肢体的标准参考位置与实际物理位置之间的平均三维欧氏距离。
  """
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Root-relative Mean Per-Keybody Position Error (R-MPKPE).

  计算相对根节点(R-MPKPE)的平均位置误差。
  该指标通过将所有肢体坐标转化为相对于根节点(即锚点)的局部坐标，
  从而剔除了全局漂移的影响, 纯粹评估当前姿势(Pose)与参考目标有多像。
  """
  # 计算参考轨迹下的各部位相对于“参考锚点”的局部坐标
  ref_anchor_pos = command.anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  ref_rel_pos = command.body_pos_w - ref_anchor_pos  # (num_envs, num_bodies, 3)

  # 计算机器人真实状态下各部位相对于“自身锚点”的局部坐标
  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  robot_rel_pos = (
    command.robot_body_pos_w - robot_anchor_pos
  )  # (num_envs, num_bodies, 3)

  # 比较两者的相对姿态误差
  pos_error = ref_rel_pos - robot_rel_pos
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Compute average joint velocity error.
  
  计算平均关节速度误差（评估动作的连贯性以及与参考速度的匹配程度）。
  """
  vel_error = command.joint_vel - command.robot_joint_vel
  return torch.norm(vel_error, dim=-1)  # (num_envs,)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector position error.
  
  计算末端执行器 (End Effector, 如手腕或足端) 在绝对空间下的对齐位置误差。
  """
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector orientation error.
  
  计算末端执行器的姿态/朝向误差（利用四元数直接计算旋转差）。
  """
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  """Get indices of specified bodies within the command's body list.

  Args:
    command: The motion command.
    body_names: Names of bodies to find.

  Returns:
    List of indices into command.cfg.body_names.
  """
  return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
