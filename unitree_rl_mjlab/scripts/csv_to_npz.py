from typing import Any

import numpy as np
import torch
import tyro
import os
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    # 保存输入和输出的参数配置
    self.motion_file = motion_file
    self.input_fps = input_fps # 动捕原始帧率（如 30 Hz）
    self.output_fps = output_fps # 物理仿真所需帧率（如 50 Hz）
    # 计算单帧之间的时间间隔 dtype: float
    self.input_dt = 1.0 / self.input_fps 
    self.output_dt = 1.0 / self.output_fps
    
    self.current_idx = 0 # 记录回放当前所处帧的索引
    self.device = device
    self.line_range = line_range
    
    # 依次执行三大步骤：加载CSV文件数据 -> 匹配输出帧率进行重采样插值 -> 对插值后位置和姿态求导算速度
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    # shape: (N, 3), 记录运动序列每一帧基座的输入空间位置 (x, y, z)
    self.motion_base_poss_input = motion[:, :3]
    # shape: (N, 4), 记录运动序列每一帧基座的输入方向四元数
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz (N, 4)
    # shape: (N, num_dofs), 记录运动序列每一帧个关节的输入角度位置。例如 g1 情况下对应的 num_dofs=29
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    # 构造输出帧的时间序列：从0到持续时间，按目标的输出帧间隔 dt 增长
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0] # 计算出插值后的总帧数 M
    
    # 依据给定的时间点序列找出两侧的输入帧索引 (index_0, index_1) 以及各自的偏置权重 blend
    index_0, index_1, blend = self._compute_frame_blend(times)
    
    # 获取输出时间点处的中心基座插值位置：对原始位置进行普通线性插值
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    # 获取输出时间点处的中心姿态四元数：对四元数进行球面线性插值(slerp)，保证插值后仍然是有效的单位四元数
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    # 针对各个物理关节随间变化的插值：线性插值
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration # 计算当前时刻在整个序列中的进度阶段(0~1)
    
    # 进度阶段 * 原始区间的数量 = 具体落在了带有小数点的原始帧数，利用 floor 取整得到左侧关键帧索序号
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    # 限制越界操作，计算得到右侧关键帧索引号
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    
    # 最后相减得到与左侧点的小数偏移比例，即插值混合时的权重 blend
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    # shape: (M, 3), 对输出帧(设M帧)的位置差分得到线速度 (vx, vy, vz)
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    # shape: (M, num_dofs), 对每个输出帧的各个关节位置求导，获得各个关节的角速度
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    # shape: (M, 3), 基于SO(3)旋转求导计算输出帧各个时刻的基座角速度 (wx, wy, wz)
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.
       给离散朝向做四元数对准，再转换到轴角形式求得每帧的 3D 绝对角速度差分算法
    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    # 切开序列获取一前一后的四元数
    q_prev, q_next = rotations[:-2], rotations[2:]
    
    # 计算两个相邻四元数的相对旋转增量： q_rel = q_next * q_prev_conjugate
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    # 转化为轴角表示，除以 2*dt 实际上得到在两个 dt 步长内的平均角速度
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    
    # 为了跟输入序列(B帧)强行对齐，复制收尾的一帧速度作为补齐
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    # 基于内部索引游标 (self.current_idx) 切出单一刻的运动学状态，均保存原始批次维度 (1, dim)
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    # 状态输出后索引向前增加一帧
    self.current_idx += 1
    
    # 判断是否到达所有的输出总帧数，到了则把索引置0允许循环，并在第二位返回终止标志 (reset_flag=True)
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_path,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
):
  # 实例化并触发完整的重采样动作数据序列加载，包含预处理所有的差分和插值操作
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  # 从场景树中定位并取出 robot 实体对象
  robot: Entity = scene["robot"]
  # 由于动作文件里的关节排序未必等同于MuJoCo模型内的排序，这里利用 joint_names 精准获取该机器人内部的索引映射
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # 准备并初始化空列表的记录字典对象，用于最后输出 NPZ
  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    # root_states shape: (num_envs, 13), 存放基底的状态 [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    # 其中 num_envs 在此配置中通常为 1
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos # 基座绝对位置
    root_states[:, :2] += scene.env_origins[:, :2] # 针对多个平行环境平移xy
    root_states[:, 3:7] = motion_base_rot # 基座的姿态 (w, x, y, z)
    root_states[:, 7:10] = motion_base_lin_vel # 基座的线速度 (vx, vy, vz)
    root_states[:, 10:] = motion_base_ang_vel # 基座的角速度 (wx, wy, wz)
    robot.write_root_state_to_sim(root_states)

    # joint_pos / joint_vel shape: (num_envs, num_dofs), 分别存放每个物理关节的位置和速度。
    # 针对 g1 机器人来说是对应的29个舵机自由度数据。
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    # 根据前面计算出的 timestep，更新环境和一些观察/控制器的物理步进状态
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    # 从物理核心引擎把更新完毕的状态全部提取出来，转储到 cpu()->numpy() 里
    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      # 取出机器人基座(body_link_pos_w 等等是指躯干在世界坐标下的姿态)的实时空间世界坐标系信息
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      # 判断引擎正向动力学解算计算的线速度/角速度值与刚才写入的目标速度之间是否有过大的非预期误差（作为检验点）
      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        np.savez(output_path, **log)  # type: ignore[arg-type]

        if render and len(frames) > 0:
          import imageio
          video_path = output_path.replace(".npz", ".mp4")
          print(f"Saving visualization video to {video_path}...")
          imageio.mimsave(video_path, frames, fps=output_fps)
          print("Video saved successfully!")


def main(
  robot: str,
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
  """
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps # 将MuJoCo原本默认的内部积分仿真时间步长(timestep)强制设为与输出帧率相同
  # 根据入参确定导入哪个机器人版本的配置：
  if robot == "g1":    # 29 Dof
    # 创建 G1 主版本的强化学习训练场景载体
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    # 按顺序定义这29个执行舵机关节的规范名字
    joint_names=[
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]
    output_dir = "./src/assets/motions/g1"
  elif robot == "g1_23dof":
    # 针对缺少手部舵机的阉割版本创建场景
    scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
    joint_names=[    # 23 Dof
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
    ]
    output_dir = "./src/assets/motions/g1_23dof"
  else:
    raise ValueError(f"Unsupported robot: {robot}")

  # 利用定义的 xml 及参数编译为真正的 mujoco 原生结构 model
  model = scene.compile()

  # 构建仿真核心对象实例，只配置1个环境 num_envs=1，因为只需用来映射和转换单一运动轨迹文件
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  # 根据刚刚拿到的模型将实体数据(如robot)填充激活
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
      entity_name="robot" # 将跟踪目标放入 viewer_cfg 中
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()
    
  os.makedirs(output_dir, exist_ok=True)
  if not output_name.endswith(".npz"):
    output_name += ".npz"
  # 拼装转换结果应写入的绝对或相对目标路径
  output_path = os.path.join(output_dir, output_name)

  # 开始正式启动逐帧仿真和数据转换收集流程
  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_path=output_path,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
