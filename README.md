# DiffSinger MAMBA (SSM Edition)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/openvpi/DiffSinger/blob/main/LICENSE)

基于 [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) 的增强分支，将 Transformer Encoder 替换为 **Mamba3 SSM (State Space Model)** + **可选混合 Transformer Self-Attention** 架构，在保持高音质的同时实现更快的 GPU 推理和更紧凑的 ONNX 模型。

## 与原始版本的差异

| 模块 | 原始版本 | Chaos SSM 版本 |
|:---|:---|:---|
| **Encoder** | Transformer (Self-Attention) | MambaEncoder（支持纯 Mamba / 混合 Attention） |
| **Variance Backbone** | WaveNet | WaveNet（保持原样） |
| **Acoustic Backbone** | LYNXNet (CNN) | LYNXNet（保持原样，可选 MambaBackbone） |
| **LR 调度** | StepLR (每 10k 步骤降 25%) | CosineAnnealingLR（平滑衰减，SSM 友好） |
| **ONNX 导出** | 标准导出 | CHUNK=128 优化 + onnxsim，节点数减少 20 倍 |
| **预处理** | 自动检测 GPU | 强制 CPU（避免多进程 CUDA 冲突） |
| **训练** | 支持 GPU | mamba-ssm CUDA kernel（GPU 训练） |

## 架构说明

### Encoder：混合 Mamba3 + Attention

采用 MambaEncoder 替代传统的 Transformer Encoder，支持通过配置控制每层类型：

- **纯 Mamba** (`layer_types: null`)：全部 BiMambaBlock（双向 SSM + GatedMambaFFN）
- **混合模式** (`layer_types: ['mamba', 'mamba', 'attention', 'attention']`)：
  - 前 N 层：**BiMambaBlock**（双向 Mamba3 SSM，局部序列建模）
  - 后 N 层：**BiMambaAttnWrapper**（标准 Transformer Self-Attention，全局语义聚合）
- **向前兼容**：不配置 `encoder_layer_types` 或留空时，行为与纯 MambaEncoder 完全一致

#### 混合模式动机

Mamba3 SSM 对 LR 骤降极度敏感（状态转移矩阵隐藏状态漂移），而 Transformer Self-Attention 在低 LR 下仍能稳定传播全局信号。在后几层使用 Attention 可以有效对抗 SSM 在低 LR 下的性能退化。

### ONNX 导出：CHUNK 优化

导出时将 BiMambaBlock 转为 **SimpleSSM**（纯 PyTorch 算子实现，无自定义 CUDA kernel），并通过 `CHUNK=128` 参数一次性处理全部 128 个状态维度：

- **CHUNK=1**：~1,280 ONNX 节点/层（20 层 ≈ 25,000 节点）
- **CHUNK=128**：~63 ONNX 节点/层（20 层 ≈ 1,260 节点）→ **减少 95%**

结合 `onnxsim.simplify` 后，完整声码器模型可在 OpenUtau 中实现流畅实时渲染。

### 预处理：强制 CPU

预处理（binarizer）阶段统一使用 CPU：

- 避免多进程 CUDA 初始化冲突
- 特征提取（librosa/pyworld/parselmouth）本身是 CPU 库
- 不受配置文件中 GPU 选项影响

### LR 调度：CosineAnnealingLR

本分支使用 `CosineAnnealingLR` 替代原版的 `StepLR`，从 6e-4 平滑衰减到 1e-5：

```yaml
lr_scheduler_args:
  scheduler_cls: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 160000
  eta_min: 0.00001
```

SSM 对 LR 骤降敏感（StepLR 每 10000 步 25% 衰减会导致验证 loss 尖峰），余弦调度比阶梯调度更适合 Mamba 训练。

### 防过拟合配置

相比原版，本分支新增以下训练正则化措施（针对小数据集 4 人 2h+ 场景）：

| 措施 | 配置项 | 值 | 说明 |
|:---|:---|:---|:---|
| Backbone Dropout | `backbone_args.dropout_rate` | 0.1 | LYNXNet 残差块内 dropout（原版 0.0） |
| L2 权重衰减 | `optimizer_args.weight_decay` | 1e-5 | 微量 L2 正则，抑制参数漂移 |
| 梯度累积 | `accumulate_grad_batches` | 2 | 增加有效 batch size 的同时加入梯度噪声平滑 |
| CosineAnnealingLR | `lr_scheduler_args` | T_max=160k, eta_min=1e-5 | 平滑衰减避免 SSM 在低 LR 下震荡 |

## 快速开始

### 环境准备

```bash
# 训练环境（需要 mamba-ssm CUDA kernel）
pip install -r requirements.txt

# ONNX 导出环境（不需要 mamba-ssm，使用 SimpleSSM 回退）
pip install -r requirements-onnx.txt
```

### 预处理数据

编辑 `data/config_acoustic.yaml` 和 `data/config_variance.yaml`，配置数据集路径后运行：

```bash
python scripts/binarize.py --config data/config_acoustic.yaml
python scripts/binarize.py --config data/config_variance.yaml
```

### 训练

```bash
# Acoustic 模型（混合 Encoder + LYNXNet Diffusion + CosineAnnealingLR）
python scripts/train.py acoustic --config data/config_acoustic.yaml --exp-name aco_testssm

# Variance 模型（WaveNet，保留原版 LR 调度）
python scripts/train.py acoustic --config data/config_variance.yaml --exp-name var_testssm
```

> **注意**：`acoustic` 子命令是 argparse 兼容标记，不影响实际参数解析。训练使用 `--exp-name` 指定实验名称，checkpoint 保存在 `experiments/<exp_name>/`。

### ONNX 导出

```bash
python scripts/export.py --config data/config_acoustic.yaml

# 导出 Variance ONNX
python scripts/export.py --config data/config_variance.yaml
```

### 部署到 OpenUtau

将导出的 `.onnx` 文件放入 OpenUtau 的 Singers 目录，配合 `dsconfig.yaml` 使用。

## 项目结构

```
├── data/                    # 数据集配置与原始数据
├── configs/                 # 通用配置文件模板
├── modules/
│   ├── commons/ssm_layers.py   # SimpleSSM, BiMambaBlock, BiMambaAttnWrapper, MambaEncoder
│   ├── backbones/              # LYNXNet, WaveNet, MambaBackbone
│   ├── fastspeech/             # FastSpeech2 encoder/variances
│   ├── pe/rmvpe/               # RMVPE 音高提取器
│   └── core/                   # DDPM / Reflow diffusion
├── preprocessing/           # Acoustic/Variance 数据预处理
├── scripts/
│   ├── binarize.py          # 预处理入口
│   ├── train.py             # 训练入口
│   └── export.py            # ONNX 导出入口
├── deployment/              # 推理和导出模块
├── training/                # 训练任务（Acoustic/Variance）
├── utils/                   # 工具函数
└── tests/                   # 测试脚本
```

## 配置参数说明

关键参数位于 `data/config_*.yaml`：

| 参数 | 说明 | 推荐值 |
|:---|:---|:---|
| `encoder_layer_types` | Encoder 每层类型 | `['mamba','mamba','attention','attention']`（混合）或 `[]`（纯 Mamba） |
| `backbone_type` (acoustic) | Diffusion 骨干网络 | `lynxnet` |
| `backbone_type` (variance) | 方差预测骨干网络 | `wavenet` |
| `lr_scheduler_args` | LR 调度器 | `CosineAnnealingLR(T_max=160000, eta_min=1e-5)` |
| `max_updates` | 总训练步数 | `160000` |
| `use_melody_encoder` | 旋律编码器（多语种/多说话人建议开启） | `true` |
| `use_glide_embed` | 滑音嵌入（需要训练数据标注） | `false` |

## 已知限制

- **mamba-ssm 与 ONNX 导出不兼容**：训练时使用 mamba-ssm CUDA kernel，导出时必须创建新环境（改用了 SimpleSSM）
- **Variance 模型不包含 SSM**：Encoder 部分使用标准 FastSpeech2，不需要 CHUNK 优化
- **CHUNK 值为模块级常量**：修改后需要重新导出 ONNX
- **混合 Encoder 与纯 Mamba checkpoint 不兼容**：切换 encoder_layer_types 后必须从头训练

## References

### 原始项目

- Paper: [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
- Implementation: [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)

### SSM / Mamba

- Mamba: [Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Mamba3 (SimpleSSM): 纯 PyTorch 选择性扫描实现，兼容 ONNX 导出

### 其他依赖

- [OpenUTAU for DiffSinger](https://github.com/xunmengshe/OpenUtau) — 生产环境部署
- [RMVPE](https://github.com/Dream-High/RMVPE) — 音高提取
- [HiFi-GAN](https://github.com/jik876/hifi-gan) + [NSF](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) — 波形重建

## Disclaimer

任何组织或个人不得在未经本人同意的情况下，利用本仓库中的功能生成他人语音，包括但不限于政府领导人、政治人物和名人。违反此条款可能构成版权侵权。

## License

本项目基于 [Apache 2.0 License](LICENSE)。
