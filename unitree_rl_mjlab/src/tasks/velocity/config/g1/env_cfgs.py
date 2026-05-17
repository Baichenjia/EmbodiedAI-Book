"""Unitree G1 velocity environment configurations.

此模块包含了针对 Unitree G1 机器人的特定环境覆盖配置。
它会调用上层基础的 make_velocity_env_cfg，并根据 G1 的物理骨骼特性进行精准的细节重写。
"""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration.
  创建 Unitree G1 在粗糙地形下的速度追踪任务配置。
  """
  # INFO: 首先获取由 make_velocity_env_cfg 生成的通用速度追踪环境模板
  cfg = make_velocity_env_cfg()

  # INFO: 针对 G1 复杂的身体多自由度接触，微调底层 MuJoCo 物理求解器的超参数以防御穿模问题
  cfg.sim.mujoco.ccd_iterations = 500       # INFO: 增加 CCD(连续碰撞检测) 迭代次数，防止高速踢踏下的穿模
  cfg.sim.contact_sensor_maxmatch = 500     # INFO: 增加传感器最大接触匹配数，适应多面体非规则地形的边界
  cfg.sim.nconmax = 48                      # INFO: 限制最大碰撞接触约束对数量

  # INFO: 将通用配置中的 "robot" 实体对象实体化为具体的 G1 配置 (导入机器人的 urdf/xml 几何资产)
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  # INFO: 把前文通用配置中 base frame 名称留空的 terrain_scan（雷达传感器），定点挂载到 G1 的 "pelvis" (骨盆/基座) 上
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  # INFO: 显式定义双足脚底的关键点 (Site) 名称
  site_names = ("left_foot", "right_foot")
  # INFO: 动态生成双脚表面所有的网格碰撞体名称，每只脚包含7 个细分的 collision 面
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # INFO: 实例化并特化配置脚部接触大地时的受力传感器 (Contact Sensor)
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      # INFO: 利用正则表达式抓取以 ankle_roll_link (踝关节翻滚节) 为树根向下的所有几何部件，视其为“脚”
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"), # INFO: 接触方靶定为 terrain(地表)
    fields=("found", "force"),  # INFO: 我们关心是否建立接触，以及确切受力向量
    reduce="netforce",          # INFO: 对这几十个部位的受力加总求取一个合力
    num_slots=1,
    track_air_time=True,        # INFO: 后台启动计时器累计该部位的离开时间（为后续Critic的足底气密时长服务）
  )
  # INFO: 设定内部自碰撞探测锚点
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # INFO: 挂载特化后的传感器
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # INFO: 启动地形曲面发生的生成器并应用课程难度递进开关
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # INFO: 为 Joint PD 控制配置引入实际 G1 原厂预设的具体动作缩放比例 (即 Action Scale)，对正归一化
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  # INFO: 为图形化监视器的目标凝视点设定为机身胸腔 "torso_link"
  cfg.viewer.body_name = "torso_link"

  # INFO: 将代表期望方向的三维虚拟箭头高度调高至上方 1.15 米悬停，方便人眼观察下发的指令方向
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # INFO: 填补之前的占位符，告知 Critic 用上述的 "site_names" 获取足底离地高度
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # INFO: 让脚底下的随机域（模拟结冰）事件能精确命中所有从脚下生成的几何构件名
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # INFO: 将模拟被踢或载重偏离事件指派给 "torso_link"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values (关于软约束与惩罚宽限 std 分配原则):
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride. (膝盖和髋部俯仰给最大容限 0.5，以便跨出自然的大步幅和下蹲避障。)
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable. (髋部滚转和偏航设紧至 0.15，防止像鸭子一样左右摇摆，维系步态轴向稳定。)
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance. (为保证平衡，脚踝滚转定得很死 0.1；踝关节俯仰相对松一点以方便抬脚尖越野。)
  # - Waist roll/pitch stay tight to keep the torso upright and stable. (躯干部位的滚转俯仰卡得很严 0.1，机器人必须保持挺直尊严不能弯腰驼背。)
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking. (肩肘稍微放任以允许走动时自然摆臂借力维持角动量。)
  # - Wrists are loose (0.3) since they don't affect balance much. (手腕游离无所谓。)
  # Running values are ~1.5-2x walking values to accommodate larger motion range. (对于奔跑任务，所有的活动容错度放大约 1.5 到 2 倍。)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05} # INFO: 罚站：要求原地必须立正站好不能随风乱扭
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body (下身约束)
    r".*hip_pitch.*": 0.5,   # INFO: 迈步主力
    r".*hip_roll.*": 0.15,   
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,        # INFO: 发力屈伸部件需宽容
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,  # INFO: 接地贴面要求最严
    # Waist (腰身)
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,  # INFO: 防止脊椎侧弯折断
    r".*waist_pitch.*": 0.1, # INFO: 防止弯腰曲折
    # Arms (双臂)
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body (奔跑期约束适度放量)
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  # INFO: 将胸腹部件正式注册给各类姿态惩罚监查计算器
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  # INFO: 把刚才找出的脚本节点名称指派给踢脚、滑步奖励扣分功能项
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  # INFO: 新增项：严厉扣分，自己撞自己（如因为骨盆内收大腿卡大腿）
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  # INFO: 当部署模式 play 为 True (不作训练) 时，执行关闭所有的阻扰和噪音代码。让策略展示最干净的能力下限。
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9) # INFO: 取消20秒暴毙重置限额，能跑一年跑一年

    cfg.observations["actor"].enable_corruption = False # INFO: 移除传感器的加入白噪音扰动
    cfg.events.pop("push_robot", None) # INFO: 移除每逢几秒被随机神秘力量推搡的域随机化事件
    cfg.curriculum = {} # INFO: 测试情况下关闭难度爬坡进程
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False # INFO: 锁定地形等级
        # INFO: 评估模式地形不用生极大，生成小片5x5即可
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration.
  平整地形专版配置文件。
  """
  # INFO: 该配置继承并包含 unitree_g1_rough_env_cfg (即上一个粗糙模式的配置) 随后在其基础上删减
  cfg = unitree_g1_rough_env_cfg(play=play)

  # INFO: 平整环境物理交互压力减小，将碰撞检测精细度和迭代要求降级下压，大幅节省算力
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  # INFO: 地貌属性直接抹平为 "plane"，无需使用柏林噪声去生成凹凸建模
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  # INFO: 【关键减负】在平坦大地上用雷达扫地没有意义（结果一定为0），因此直接删减该部件！
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  # INFO: 既然没了雷达，Actor 和 Critic 这两个网络也就不需要接收这个高程输入层了，同步删除该字典参数
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  # INFO: 因为纯平大地无法逐级提升台阶或碎石难度，直接去掉地形课程等级演进器
  cfg.curriculum.pop("terrain_levels", None)

  # INFO: 当运行模式 play 启动时，对于平坦地形，将运行期间随机指令索取的速率稍加降维缩口
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
