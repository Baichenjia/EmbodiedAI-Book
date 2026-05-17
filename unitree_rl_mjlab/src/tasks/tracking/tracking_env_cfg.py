"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.tracking.mdp as mdp

# 默认全局扰动的速度范围（包含线速度 XYZ 和欧拉角速度 Roll, Pitch, Yaw）
# 这些用于控制在随机化或事件注入时，能给机器人推撞出多大的初始速度偏差
VELOCITY_RANGE = {
  "x": (-0.5, 0.5),   # 前后推力范围
  "y": (-0.5, 0.5),   # 左右推力范围
  "z": (-0.2, 0.2),   # 上下扰动范围
  "roll": (-0.52, 0.52),  # 翻滚角速度随机约束
  "pitch": (-0.52, 0.52), # 俯仰角速度随机约束
  "yaw": (-0.78, 0.78),   # 偏航角速度随机约束
}


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base tracking task configuration."""

  ##
  # Observations (输入特征空间，提供给 Actor 和 Critic 的状态观测字典)
  ##

  actor_terms = {
    # 从运动文件中取得的参考/目标状态数据
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    # 基座相对于“运动目标对齐锚点”的位移差，并加入 (-0.25, 0.25) 的均匀噪声模拟由于里程计不精确造成的估计误差
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    # 基座相对于“运动目标对齐锚点”的方向/旋转差，同样加入微小均匀旋转噪声
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # 直接由仿真内部 IMU 模拟出来的机体线速度测量值（附带噪声）
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # 仿真内部 IMU 采集到的机体角速度测量值（附带噪声）
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # 所有物理关节此时处于什么相对位移角度（加上了零偏偏差模拟，见下面的 encoder_bias）
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    # 所有物理关节的实时角速度（加噪声）
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    # 上一步(t-1时刻)强化学习网络下发给各关节的控制动作缓存
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    # Critic 属于全知者（主要用于计算 Value 而不是生成真实环境的 action），
    # 因此其接收的信息都是纯净没有 `noise` 注入的绝对真实环境状态。
    
    # 1. 纯净无干扰的任务目标控制指令
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    # 2. 真实无偏的基座相对轨迹参考坐标系的位移偏差
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    # 3. 真实无偏的基座相对轨迹参考坐标系的旋转姿态差
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    # 4. 机器人各个关键肢体部位在本体坐标系中的精准相对位置
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    # 5. 机器人各个关键肢体部位在本体坐标系中的精准相对旋转
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    # 6. 底层物理引擎计算出的绝对本体线速度（剥离了 IMU 传感器模拟的噪声和滤波延迟）
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    # 7. 底层物理引擎计算出的绝对本体角速度（无传感器随机漂移噪声）
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    # 8. 绝对真实的物理关节当前转角与位移（去除了编码器 bias 的静态注入干扰）
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    # 9. 绝对真实的物理关节瞬时角速度/线速度（剥离速度测算噪声）
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    # 10. 上一控制步下发的实际 Action 缓存，这是构建马尔可夫决策过程转移价值的关键历史凭据
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    # 将以上的特征列表拼接组成 (Batch_Size, Dim) 的张量数据，作为强化学习的最终输入状态
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True, # 对于执行者激活传感器噪声破坏，增加控制鲁棒性
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False, # 价值评估网络保持无损观察以获得最优环境价值预测
    ),
  }

  ##
  # Actions (Actor 网络生成的动作定义方式)
  ##

  actions: dict[str, ActionTermCfg] = {
    # 基于关节目标角度的 PD 位置控制器作为动作空间
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",), # 通过正则表达式匹配所有机器人的驱动节点
      # action 的直接缩放系数。因此最终目标位置 = 默认位置 + network_output * scale
      scale=0.25, 
      use_default_offset=True,
    )
  }

  ##
  # Commands (命令目标获取管理器，负责从外界提取控制和参考输入信号)
  ##

  commands: dict[str, CommandTermCfg] = {
    # 从提供的 csv_to_npz 的轨迹文件中提取指令
    "motion": MotionCommandCfg(
      entity_name="robot",
      # resampling_time_range 设的极大意味着这里强制不进行随机切换采样，而是一直循着轨迹顺序放下去
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      # 每次环境重启，会在初始轨迹的基座上给一个位置和姿态的随机均匀偏置（Domain Randomization）
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1), # 初始化关节位置时允许有小范围的随机偏移
      # 下面这些具体参数在实际跑时会被具体机器人的子配置或 CLI 命令行（如您的那句传参）覆盖
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events (仿真事件管理器，用于做环境物理特性的扰动来填补 Sim2Real Gap)
  ##

  events: dict[str, EventTermCfg] = {
    # 在每间隔 1.0s 到 3.0s 之间的某刻中，突然重置机器人各个躯干部分的速度，效果相当于受到了外界猛推
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    # 系统重置(startup)或开头时，将其身体躯干中心(CoM)进行小范围的质心偏移，模拟实体机器人加工与设计误差
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    # 模拟由于编码器归零不到位或物理磨损导致的角度偏移误差，在启动时给每个关节加上绝对的常数静态误差
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    # 改变各个地形块对机器人的鞋面能提供的摩擦力，覆盖低摩擦的冰面到粗糙的路面以适应任何质地
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Rewards (奖励系统，引导代理优化的方向与标准)
  ##

  rewards: dict[str, RewardTermCfg] = {
    # 基于高斯核差分，当机器人位于世界系下正确的锚点追踪位置附上正反馈
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 奖励肢体与基座坐标系之间的相对坐标与指令给出的参考目标接近
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    # 奖励肢体与基座坐标系之间的相对旋转姿态近似情况
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 奖励躯干中心移动速度（世界系下的线速度与参考匹配）
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    # 奖励本体角速度在各个维度上的收敛逼近
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # [负反馈] 惩罚相邻步骤之间输出差距过大（鼓励平滑连续的输入，防止高频震荡或者关节猛烈抽动）
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    # [负反馈] 严厉惩罚到达或企图越过机构物理和电机能够达到的行程上下限的姿势
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # [负反馈] 当测探器检测到自身肢体部位相撞交叠的力超过 10 时，处以严厉罚款
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations (任务强制终止规则，避免智能体进入无意义区域继续探索)
  ##

  terminations: dict[str, TerminationTermCfg] = {
    # 每个 episode 跑满了指定的时间(或步数)时安全终止，给一局画上句号。`time_out=True` 表示这不是失败导致的终止，通常算做成功。
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # 失败判断：仅比较物理系统 Z 轴(高度)跟期望运动的 Z 轴之间如果偏离了 0.25，则算作彻底失去平衡（比如掉落砸向地板跌倒）
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    # 失败判断：当前自身的姿态与目标期望姿态旋转差异过大（如前倾超限）
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    # 失败判断：执行末端(比如手臂)相较于轨迹要求只在 Z 轴上高度偏离了超过 25cm
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  ##
  # Assemble and return (打包组装，构建并返回最终给引擎的核心配置字典)
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1), # 默认空场景地面，环境数 1 (后续可被外部指令如 num-envs=4096 覆盖)
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    # 默认可视化监控相机位姿跟踪视角
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    # 物理模型层仿真核心运算控制：求解器的松弛迭代次数、连接触网格配置
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4, # 降采样率：RL 神经网络策略每输出 1 步动作，底层的 mujoco 会持续计算执行 4 次细分积分即 (4 x 0.005 = 控制率 50 Hz 频率)
    episode_length_s=10.0, # 一局游戏允许的最长物理时间为 10 秒
  )
