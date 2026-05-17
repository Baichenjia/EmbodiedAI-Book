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
    env_cfg = load_env_cfg(task_id)
  # INFO: agent_cfg 提取自 cfg.agent 的字典形式，包含算法层面的超参数
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  # INFO: cuda_visible = '0'
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    # INFO: seed = 42 (根据默认随机种子决定)
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # INFO: local_rank = 0
    rank = int(os.environ.get("RANK", "0"))
    # INFO: rank = 0
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # INFO: device = 'cuda:0'
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank
    # INFO: seed = 42 (根据默认随机种子决定)

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
  # INFO: is_tracking_task = False (由于运行的是 Unitree-G1-Flat 任务)
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

  # INFO: env 为 ManagerBasedRlEnv 的实例化对象，底层集成了 4096 个并行 MuJoCo 环境
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  # INFO: resume_path = None
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

  # INFO: env 被包装进了 RslRlVecEnvWrapper 适配器对象中
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  # INFO: agent_cfg 提取自 cfg.agent 的字典形式，包含算法层面的超参数
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  # INFO: runner_cls = <class 'mjlab.rl.runners.mjlab_on_policy_runner.MjlabOnPolicyRunner'>
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)
  # INFO: runner 对象为 PPO 执行器主逻辑实例，负责统筹采样和反馈更新

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
  # INFO: args 等同于 TrainConfig(env=..., agent=..., gpu_ids=[0], ...)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  # INFO: log_root_path = PosixPath('logs/rsl_rl/g1_velocity') (示例值，取决于 agent.experiment_name)
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  # INFO: log_dir_name = '2026-05-03_XX-XX-XX'
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  # INFO: log_dir = PosixPath('logs/rsl_rl/g1_velocity/2026-05-03_XX-XX-XX')

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  # INFO: selected_gpus = [0], num_gpus = 1

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    # INFO: os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

  # INFO: all_tasks = ['Unitree-Go2-Flat', 'Unitree-G1-Flat', 'Unitree-G1-Tracking-...] 包含了所有的可用任务
  all_tasks = list_tasks()
  # INFO: chosen_task = 'Unitree-G1-Flat', remaining_args = ['--env.scene.num-envs=4096']
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # INFO: 最终的 args 包含合并后的 TrainConfig(env.scene.num_envs=4096...)
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
