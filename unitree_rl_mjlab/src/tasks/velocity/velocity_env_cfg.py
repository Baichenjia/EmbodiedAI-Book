"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.velocity.mdp as mdp


# INFO: 相比传统的重写 gym.Env 的 step() 和 reset()，mjlab 采用工厂模式。
# 所有环境配置集中在一个 ManagerBasedRlEnvCfg 配置类中实例化，底层框架会据此自动编排物理引擎的运算流程。
def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors (传感器配置)
  # 这里定义了机器人带有的各种传感器。
  # 这些传感器获取的数据将在后续的 observations 和 rewards 中被使用。
  ##

  # INFO: 雷达扫描传感器，模拟激光雷达或深度相机感知前方的地形起伏。
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    # INFO: frame 指定了该雷达挂载在哪个基准刚体上（实体名为 "robot"，具体部位名 "" 留空，由具体机器人如 g1 专属配置再覆写为 "pelvis"）
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    # INFO: 创造一个 1.6 x 1.0 尺寸的网格状射线投射（类似俯视深度图），分辨率为 0.1 米
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0, # 最大可探测 5 米
    exclude_parent_body=True, # 发射射线时忽略机器人本体的碰撞
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations (观测空间配置)
  # 采用了非对称的 Actor-Critic 架构：
  # Actor 只能看到机器人自身能获取的本体感受信息（Proprioception）。
  # Critic 训练时拥有上帝视角，可以看到额外的特权信息（Privileged Info，如精确线速度、接触力等）。
  ##

  # INFO: Actor 观测项映射：策略网络（Policy Network）在实际部署时能依赖的输入。所有 term 底层都是纯函数。
  actor_terms = {
    # INFO: 机器人的机身角速度（通过读取 IMU 传感器得出）
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2), # INFO: 在配置层直接注入 -0.2 到 0.2 的均匀噪声，模拟真实 IMU 传感器的误差
    ),
    # INFO: 投影重力向量，由于机器人是自由移动的，感知重力向量的方向能帮助它知道自己相对大地的姿态倾斜角
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # INFO: 指令管理器下发的期望移动速度 [vx, vy, yaw_rate]
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # INFO: 运动的步态相位（Phase），帮助机器人的神经网络知道当前周期 0.6 秒中进行到了哪一步（是该抬左脚还是右脚）
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    # INFO: 机器人所有马达的当前相对位置（相当于编码器反馈）
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # INFO: 关节的角速度反馈
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # INFO: 神经网络上一步输出的动作，方便网络平滑后续的控制
    "actions": ObservationTermCfg(func=mdp.last_action),
    # INFO: 雷达传入的周边高程图地形数据，告知前方有无台阶/坑洼
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance, # 归一化输入
    ),
  }

  # INFO: Critic 观测项映射：价值网络（Value Network）在仿真训练时额外输入的上帝视角特权信息
  critic_terms = {
    **actor_terms, # INFO: Critic 包含了 Actor 的全部观测输入
    # INFO: 特权信息：真实的底盘绝对线速度。但在现实中，由于没有精准定位，机器人本体（Actor）测不准这个值
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # INFO: 无加噪的真实高度地图信息
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    # INFO: 当前双脚真实的对地高度（用于 Critic 价值评估）
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    # INFO: 脚部滞空时间（由接触传感器解析得出，仅 Critic 可见）
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # INFO: 脚部真实接触状态和精准的受力向量
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    # INFO: Actor 组，允许加入噪声 corruption，历史长度历史帧数为1
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    # INFO: Critic 组，作为基准价值判断，不加人为误差干扰
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    # INFO: 评估运行性能：平滑度（均方动作加速度指标）
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions (动作空间配置)
  # 定义网络输出的目标信号模式
  ##

  actions: dict[str, ActionTermCfg] = {
    # INFO: 声明底层的动作控制器为“节点位置”（PD控制器）。
    # 策略网络的输出将被乘上 scale=0.25 (可由具体机器人覆写，例如对于G1可能会调整)，并叠加默认站立偏离角。
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands (指令管理器配置)
  # 定义任务目标。这里的任务是追踪期望速度 (Twist Command)。
  # 会定期重新采样一段期望的前进/侧向/转向速度。
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0), # INFO: 在 3 到 8 秒间随机决定刷新下一次指令的时间点。
      rel_standing_envs=0.05, # INFO: 5% 的环境会抽到全0指令（站立不动任务）
      heading_command=True, # 开启朝位指令计算
      heading_control_stiffness=0.5,
      debug_vis=True,
      # INFO: 速度采样上下限：如 X 轴线速度 [-1.0 到 2.0 m/s]
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events (事件与域随机化 Domain Randomization配置)
  # 通过在不同的仿真时机(mode)触发纯函数，增加环境噪声，提高 Sim-to-Real 的鲁棒性。
  ##

  events = {
    # INFO: mode="reset"。当某个环境由于超时或跌倒重置时，随机出生在一个位置姿态（如偏航角 yaw 随机）
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    # INFO: mode="reset"。重置环境时，同时将各关节恢复并附加轻微扰动。
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # INFO: mode="interval"。固定周期（每 5.0 到 6.0 秒）触发一次机器人的外界推力。
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": { # INFO: 这相当于突然在 x,y,z 以及各个滚转角上加一个瞬时速度突变（模仿被踢了一脚）
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    # INFO: mode="startup"。在整个仿真启动时只运行一次：随机化机器人脚与地面的接触摩擦系数（0.3到1.6），让步态学会适应滑冰场和柏油路。
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    # INFO: mode="startup"。启动时模拟电机编码器的安装偏置误差。
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # INFO: mode="startup"。启动时模拟质心（重心）因负载或安装造成的偏移误差。
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards (奖励配置)
  # 定义了 MDP 的奖励函数项。weight > 0 为奖励，weight < 0 为惩罚。
  # 框架会自动将所有奖励项在每一步求和，作为强化学习的运行回报。
  ##

  rewards = {
    # INFO: 奖励：鼓励实际本体线性速度跟随 command 下发的指令速度（权重 1.0）
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # INFO: 奖励：鼓励跟随角速度指令
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # INFO: 惩罚：主躯干的三维姿态如果偏离水平或者直立设定，就会严厉扣分 (保证机器人不要东倒西歪，权重 -1.0)
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    # INFO: 奖励（带惩罚特性）：控制每个节点相对其默认姿势（Reference Posture）的游离程度。
    # params 内部会分配如 std_walking, std_running 等软边界。超界扣分。
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    # INFO: 惩罚：主躯干多余的滚转摇晃角速度越界惩罚（保证运动平顺性）
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    # INFO: 惩罚：系统全局角动量过大（防止双臂像大风车一样乱甩来维持平衡）
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    # INFO: 惩罚：如果环境因为摔倒等因素触发 terminate 终止条件，扣除致命的 -200 分！
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # INFO: 惩罚：关节加速度的二范数惩罚。加速度过大意味着输出力矩频繁突跳伤电机，权重极小 (-2.5e-7) 去做平滑。
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    # INFO: 惩罚：严惩关节位置超过设计的物理机械极限限制。 (-10.0)
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # INFO: 惩罚：动作变化率惩罚，希望相邻帧网络输出的策略量尽可能连续。 (-0.05)
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    # INFO: 奖励：诱导周期步态节奏。要求在当前时间相位(Phase)下应该着地的脚接触地面，应该抬起的脚离开地面。
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6, # INFO: 步频周期设置为 0.6秒
        "offset": [0.0, 0.5], # INFO: 左脚和右脚启动相位错开半个周期
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact", # INFO: 利用前文传感器数据监测真伪
      }
    ),
    # INFO: 惩罚：脚底板抬起时如果没有离地 10厘米 (target_height) 以上导致可能被小石头绊倒，给予扣分。
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # INFO: 惩罚：脚底虽然落地了结果还在地上滑出一段距离（摩擦力建模导致），扣分。
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # INFO: 惩罚：落地砸地瞬间的 F_z 法向阻力极大，要求它落地尽量温柔。
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # INFO: 惩罚：如果下发的指令 twist 几乎为 0，你就不准动，老老实实站好。
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }

  ##
  # Terminations (终止条件配置)
  # 定义什么情况下 Episode 会被提前阻断重置 (Reset)。
  ##

  terminations = {
    # INFO: 正常的时间达到上限超时（不是死亡，会重置但可能不扣 200分）。
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # INFO: 发生严重翻倒事故：机身倾角达到或超过 70度 时，触发摔倒死亡重置并严厉扣分。
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum (课程学习配置)
  # 强化学习直接学习高速跑和越野太难了，这里通过步骤进度循序渐进地提升目标难度。
  ##

  curriculum = {
    # INFO: 当地形适应良好时，程序生成的地形复杂度（地形等级 Terrain Level）逐步增加。
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # INFO: 当训练处于第 0 步时，先教它跑慢速任务（X轴线速度最高1.0）。
    # 等到步数达标（5000*24步）后，逐渐放宽指令给到极速状态（X轴允许高达2.0）。
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return (组装所有 Managers 并返回环境配置)
  # 将上面配好的拼图交给引擎调度核心，形成一个能交给 PPO 或者其它 RL 算法读取的主环境参数。
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator", # INFO: 指示底层引擎通过生成器动态刷新地形（而不是读取固定的平面 mesh）
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1, # INFO: 并行环境数（这里的 1 在外层被传参 `--env.scene.num-envs=4096` 所覆盖重写）
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg( # INFO: 指定底层跑 MuJoCo 物理引擎，物理计算帧步长 5毫秒
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4, # INFO: RL 策略的控制频率决策周期。5ms * 4 = 20ms，即策略网络每 20ms(即50Hz) 输出一次控制动作。
    episode_length_s=20.0, # INFO: 单个回合最长存活 20 秒，如果不死就会在此之后触发 time_out 结算并无伤重置以搜集其它轨迹。
  )
