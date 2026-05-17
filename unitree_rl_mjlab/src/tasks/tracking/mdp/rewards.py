from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  """内部辅助函数：根据提供的 body_names 列表，从预先加载的追踪目标节点集合中提取对应的张量索引号。"""
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """计算锚点（通常是机器人的根节点或躯干）在全局世界坐标系下的位置吻合度奖励。
  
  机制：使用高斯核（负指数衰减函数） exp(-error / std^2)，将无限大的欧氏距离平方误差转化为 (0, 1] 之间的连续平滑奖励。
  距离越近得分越趋近于 1，偏离越远（超过 std 容差后）得分急剧衰减至 0。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """计算锚点在全局坐标系下的姿态/朝向对齐奖励。
  
  机制：通过计算参考序列的目标四元数与仿真物理四元数之间的误差并求平方，再经由指数映射为 [0,1] 奖励。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """计算各个肢体（如手腕、脚踝）相对于自身锚点（躯干）的局部相对位置重合度奖励。
  
  机制：纯粹评估当前动作“做得像不像预期的内部空间骨架结构”。
  此方式切断了世界坐标的漂移干扰，即便机器人整体走偏，只要身姿与参考序列保持一致，此处仍给高分。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """计算各个评估肢体节点相对于本体锚点的局部相对旋转/姿态奖励。
  
  机制：保障像末端执行器这样的局部节点，虽然位置上符合了，但其细微的扭曲朝向也能完美贴合。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """计算物理肢体节点在全局空间中的线速度的一致度奖励。
  
  机制：不但要求“某一帧你处于这个位置”，还要求你在追踪该轨迹时“正保持着与轨迹同步的运行速度矢量”。
  有利于产生平滑、连贯无突变的跟随动画，保证其动态合乎专家示教的动量规律。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """计算机器人的追踪肢体部位与目标的角速度矢量的一致度奖励。
  
  机制：减少即便身体姿态已经拟合，但是由于控制器高速震荡导致的“小幅高频抽搐”瑕疵。通过对齐角速度抑制这种异常输出。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions. （对自身穿模或四肢相碰的行为施以负向惩罚的代价函数）

  机制：在追踪高难度动捕动作时，模型极容易扭曲自己导致左右脚等互相绊倒或互相嵌合（穿模）。
  本函数在后台统计 `history_length > 0` 范围内的多次物理微积分步里，一旦任何一个相撞的接触力张量（欧氏范数）爆表，
  超过了阈值 (如 10.0 N)，这就会直接累计违规次数，作为扣分传递给 PPO，以此引导策略学习懂得去寻找物理空间可执行路径。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)