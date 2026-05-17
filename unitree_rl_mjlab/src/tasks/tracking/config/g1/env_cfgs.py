"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  # 1. 继承并获取基础通用的 tracking_env_cfg 配置字典
  cfg = make_tracking_env_cfg()

  # 2. 实体更新：将泛用的 robot 名称绑定为具体的 Unitree G1 实体模型配置
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # 3. 碰撞传感器更新：为了实现 self_collision 惩罚，在这里显式定义了骨盆等相关网格之间的自碰撞检测器
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # 4. 动作缩放更新：替换默认的 0.25 统一缩放，改用专属于 G1 机器人各关节的动态 action_scale（适配不同电机的力矩与活动范围）
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  # 5. 轨迹追踪参考部位更新：覆盖 tracking_env_cfg 空的 () 配置，注入 G1 的专属 14 个核心追踪节点（髋、膝、踝、躯干、肩、肘、腕）
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link" # 系统锚点设定为躯干
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  # 6. Domain Randomization(域随机化) 对象特化指向
  # 针对地脚摩擦力随机化，精准指向左脚和右脚一共14个碰撞检测点
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  # 重心偏移扰动注入点：指向躯干 torso_link
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # 7. 失败终止条件靶点更新：设置末端追踪容差节点为 G1 的最终四肢末端（脚踝和手腕）
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  # 8. 默认相机追踪的焦点改为锁定躯干部位
  cfg.viewer.body_name = "torso_link"

  # 9. [功能覆写] 绝对状态估计剥离：如果要在物理实体上部署(无法通过外设得到绝对世界坐标和线速度估计)
  # 那么会从 Actor 的观测向量中把 `motion_anchor_pos_b` 和 `base_lin_vel` 彻底剔除。
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True, # 真实机器人上其它传感器依然有噪声
    )

  # 10. [功能覆写] 播放模式 (Play Mode)：当您只想用之前训练好的 Checkpoint 来可视化看效果时
  if play:
    # 让一局的时间变成极限大，除非摔倒否则一直跑
    cfg.episode_length_s = int(1e9)

    # 关闭所有的训练环境噪音（观测噪声清零、撤销重外力推搡）
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # 关闭参考状态初始化 (RSI) 的一切位姿随机扰动（直接从目标序列完美起手）
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    # 从头开始播放目标序列，而不是在训练集随意中间采样截断
    motion_cmd.sampling_mode = "start"

  return cfg
