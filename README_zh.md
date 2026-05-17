# EmbodiedAI-Book 工作区总览（中文）

本目录包含两个主要仓库：

- lerobot：偏向通用具身智能模型、数据集与机器人策略训练/推理框架。
- unitree_rl_mjlab：偏向 Unitree 机器人在 MuJoCo 中的强化学习训练、仿真验证与实机部署。

## 两个 Repo 的作用

| 维度 | lerobot | unitree_rl_mjlab |
|---|---|---|
| 核心目标 | 通用机器人学习框架（数据、策略、训练、评测、部署） | 面向 Unitree 机器人的 RL 训练与部署链路 |
| 主要方法 | 模仿学习 + VLA（如 ACT、pi0、pi05、pi0_fast）+ 统一数据格式 | 强化学习（速度跟踪、动作模仿） |
| 典型输入 | LeRobotDataset（图像/视频 + 状态动作）和任务文本 | MuJoCo 仿真环境、动作文件（如 npz）、奖励函数配置 |
| 典型输出 | 可复用策略权重、评测结果、可接入真实机器人推理 | 训练好的 checkpoint、onnx 策略、可部署到 Unitree 控制程序 |
| 更适合谁 | 做策略研究、跨平台机器人任务、VLA 训练与评测 | 做 Unitree 定向控制、强化学习、Sim-to-Real 落地 |
| 使用重心 | 训练/评测命令统一，支持多策略与多硬件 | 训练 -> 仿真验证 -> 实机部署流程清晰 |

## 仓库结构

- lerobot/
- unitree_rl_mjlab/

建议先分别阅读：

- lerobot/README_zh.md
- unitree_rl_mjlab/README_zh.md

---

## 1) lerobot：作用与使用方式

### 1.1 这个仓库解决什么问题

lerobot 是一个通用的机器人学习工具链，重点在于：

- 统一机器人控制接口，减少“换机器人就重写整套代码”的成本。
- 标准化数据格式（LeRobotDataset），方便采集、复用、共享与训练。
- 集成多个策略族（ACT、pi0、pi05、pi0_fast、rtc 推理增强等），支持从训练到评测再到部署。

### 1.2 什么时候优先用 lerobot

- 你要做端到端策略训练（尤其是视觉 + 语言 + 动作）。
- 你需要将模型迁移到不同机器人平台。
- 你希望使用标准化数据集进行可复现实验。
- 你要跑基准评测（例如 LIBERO）。

### 1.3 快速上手流程

1. 安装与环境确认

```bash
cd lerobot
pip install lerobot
lerobot-info
```

2. 选择策略并训练（示例：ACT）

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda
```

3. 策略评测（示例：pi0 在 LIBERO）

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

4. 若需要更平滑在线执行，可启用 RTC

```bash
lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --policy.path=your_org/your_policy \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --task="Pick and place" \
  --duration=120 \
  --device=cuda
```

### 1.4 常用策略速查

- ACT：轻量模仿学习，训练快，入门友好。
- pi0：VLA + flow matching，适合通用操作任务。
- pi0_fast：FAST 动作分词 + 自回归，训练效率高。
- pi05：pi0 增强版，强调开放场景泛化。
- rtc：推理时增强，不是独立策略，用于减少 chunk 执行不连续。

### 1.5 关键目录（便于二次开发）

- src/lerobot/policies：策略实现
- src/lerobot/datasets：数据集相关功能
- src/lerobot/robots：机器人接入层
- examples/：教程和可运行示例
- docs/source/：策略文档与概念说明

---

## 2) unitree_rl_mjlab：作用与使用方式

### 2.1 这个仓库解决什么问题

unitree_rl_mjlab 聚焦 Unitree 机器人在 MuJoCo 下的强化学习训练与部署，典型闭环是：

- 训练（多并行环境）
- 仿真回放验证
- 导出策略并部署到实机

它特别适合做速度控制、动作模仿、以及 Sim-to-Real 迁移。

### 2.2 什么时候优先用 unitree_rl_mjlab

- 目标机器人是 Unitree 系列（Go2、G1、H1_2 等）。
- 你要做强化学习训练，而不是以离线模仿学习为主。
- 你需要从 MuJoCo 快速迭代到可部署控制程序。

### 2.3 快速上手流程

1. 安装环境

```bash
cd unitree_rl_mjlab
# 按项目文档执行环境安装
# 参考：doc/setup_zh.md
```

2. 速度跟踪训练（示例）

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

3. 动作模仿训练（示例）

```bash
python scripts/train.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096
```

4. 仿真回放验证

```bash
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

5. 导出与部署

- 训练目录会产生模型与可部署资产（含 onnx）。
- 根据机器人型号在 deploy/ 下编译对应控制程序。
- 先做仿真部署，再上实机部署。

### 2.4 常用脚本

- scripts/train.py：训练入口
- scripts/play.py：回放与可视化
- scripts/csv_to_npz.py：动作文件预处理
- scripts/list_envs.py：查看可用环境

### 2.5 关键目录

- src/tasks：任务定义
- src/assets：机器人资源、动作资源
- deploy/：实机控制程序与配置
- simulate/：仿真侧程序
- logs/：训练日志与模型产物


