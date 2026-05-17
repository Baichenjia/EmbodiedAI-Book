<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** 是一个面向真实机器人场景的 PyTorch 开源库，提供统一的机器人控制接口、标准化数据集格式、以及可训练可部署的前沿策略模型，目标是降低具身智能开发门槛。

- 硬件无关、Python 原生控制接口，覆盖从低成本机械臂到人形机器人。
- 标准化 LeRobotDataset（Parquet + MP4/图片），可在 Hugging Face Hub 上高效存储、流式读取与可视化。
- 集成多类 SoTA 策略，支持训练、评测与部署闭环。

## 快速开始

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> 完整安装说明请参考 [Installation Documentation](https://huggingface.co/docs/lerobot/installation)。

英文版说明见：[README.md](./README.md)

## 机器人与控制

LeRobot 提供统一 `Robot` 接口，将上层控制逻辑与底层硬件细节解耦。你可以把任意硬件接入到统一的数据采集、训练和推理流程中。

```python
from lerobot.robots.myrobot import MyRobot

robot = MyRobot(config=...)
robot.connect()
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

## 数据集

LeRobotDataset 用于解决机器人数据格式碎片化问题。

- 视觉数据：同步 MP4 或图像
- 状态/动作数据：Parquet
- 支持删除 episode、按比例切分、特征增删、数据集合并

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")
print(dataset[0]["action"].shape)
```

更多说明： [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 模型与策略

LeRobot 的策略实现位于 [src/lerobot/policies](./src/lerobot/policies)。

### ACT（src/lerobot/policies/act）

ACT（Action Chunking with Transformers）是轻量级模仿学习策略，训练速度快、参数规模相对小，适合入门与快速验证。

训练：

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda
```

教程脚本：

```bash
python examples/tutorial/act/act_training_example.py
```

文档： [ACT](./docs/source/act.mdx)

### pi0（src/lerobot/policies/pi0）

pi0 是基于 flow matching 的 Vision-Language-Action（VLA）策略，适合语言条件下的通用机器人控制。

安装依赖：

```bash
pip install -e ".[pi]"
```

训练：

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --output_dir=outputs/train/pi0_your_dataset \
  --job_name=pi0_your_dataset \
  --policy.device=cuda
```

评测（LIBERO 示例）：

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

硬件推理示例：

```bash
python examples/tutorial/pi0/using_pi0_example.py
```

文档： [pi0](./docs/source/pi0.mdx), [policy_pi0_README](./docs/source/policy_pi0_README.md)

### pi0_fast（src/lerobot/policies/pi0_fast）

pi0_fast 采用 FAST 动作分词（频域动作序列 tokenization）进行自回归动作生成，在很多场景下相较扩散式动作生成可显著提升训练效率。

安装依赖：

```bash
pip install -e ".[pi]"
```

训练：

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=pi0_fast \
  --policy.pretrained_path=lerobot/pi0_fast_base \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.max_action_tokens=256 \
  --output_dir=outputs/train/pi0fast_your_dataset \
  --job_name=pi0fast_your_dataset \
  --policy.device=cuda
```

文档： [pi0fast](./docs/source/pi0fast.mdx)

### pi05（src/lerobot/policies/pi05）

pi05 是 pi0 的增强版本，重点提升开放场景泛化能力（open-world generalization），同样属于 VLA 系列。

安装依赖：

```bash
pip install -e ".[pi]"
```

训练：

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=outputs/train/pi05_your_dataset \
  --job_name=pi05_your_dataset \
  --policy.device=cuda
```

评测（LIBERO 示例）：

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

文档： [pi05](./docs/source/pi05.mdx), [policy_pi05_README](./docs/source/policy_pi05_README.md)

### rtc（src/lerobot/policies/rtc）

rtc（Real-Time Chunking）是推理时增强方法，不是独立策略。它用于在存在推理延迟时，让 chunk-based 流匹配策略（如 pi0、pi05）在执行上更平滑、连续。

离线评估示例：

```bash
python examples/rtc/eval_dataset.py \
  --policy.path=lerobot/pi0_libero_finetuned \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --rtc.execution_horizon=10 \
  --rtc.max_guidance_weight=10.0 \
  --device=cuda
```

实机 rollout + rtc：

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

文档： [rtc](./docs/source/rtc.mdx), [policy_rtc_README](./docs/source/policy_rtc_README.md)

## 推理与评测

LeRobot 支持在仿真和真实硬件上统一评测流程，可用于 LIBERO、MetaWorld 等基准。

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

## 资源

- [Documentation](https://huggingface.co/docs/lerobot/index)
- [Discord](https://discord.gg/q8Dzzpym3f)
- [X](https://x.com/LeRobotHF)
- [Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)

## 引用

如果你在项目中使用 LeRobot，请参考英文 README 中的 BibTeX 进行引用： [README.md](./README.md)

## 贡献

欢迎通过 issue / PR 参与共建。开始前建议阅读 [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md)。
