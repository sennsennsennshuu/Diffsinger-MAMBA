# DiffSinger-SSM

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

基于 [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) 的分支，把 Encoder 换成 Mamba3 SSM（可与 Self-Attention 混合），并在此基础上做了 ONNX 部署修复、TuneLab 引擎扩展和推理速度优化。

仓库根目录是 Python 训练 / 导出代码；`extensions/ChoristaDiffsinger-SSM/` 是为 TuneLab v1.6.0 写的 .NET 8 引擎扩展（产物 `Diffsinger-SSM.tlx`）。

---

## 目录布局

```
modules/commons/ssm_layers.py     SimpleSSM（ONNX 用 fallback，已修 A_neg 算符优先级 bug）
modules/fastspeech/               MambaEncoder + Mamba3/Self-Attention 混合层
modules/backbones/                扩散主干：lynxnet / wavenet / mamba（aco 默认 lynxnet）
modules/core/reflow.py            Rectified Flow 采样器：euler / rk2 / rk4 / rk5
deployment/exporters/             ckpt → ONNX 导出
configs/templates/                训练配置模板
tools/                            云端 parity / profile / artifact 同步等运维脚本
extensions/ChoristaDiffsinger-SSM 独立 TuneLab 引擎，输出 Diffsinger-SSM.tlx
docs/SSM_SPEC.md                  SelectiveScan 数学规范与等价性说明
```

## 相对上游的核心改动

- Encoder：Transformer → MambaEncoder（BiMambaBlock × N），可逐层混入 Self-Attention
- LR 调度：StepLR → CosineAnnealingLR（SSM 在阶梯下降下 loss 容易尖峰）
- 正则化：dropout 0.0→0.1，加 weight_decay，调大梯度累积
- ONNX 部署：CUDA mamba_ssm 算子无法导出 → SimpleSSM 等价回退；修复 `A_neg` 的算符优先级 bug（`(-softplus(...)).clamp(max=-1e-4)`，此前未加括号导致 ADT≈0、合成"含糊"）
- TuneLab 引擎：独立 `Diffsinger-SSM.tlx`，与 ChoristaDiffsinger 共存不冲突

## 训练 / 导出

```bash
python scripts/binarize.py --config data/config_acoustic.yaml
python scripts/train.py acoustic --config data/config_acoustic.yaml --exp-name aco_xxx
python scripts/train.py variance --config data/config_variance.yaml --exp-name var_xxx
python scripts/export.py --exp aco_xxx
```

Encoder 模式（`encoder_layer_types`）：

| 模式 | 配置 | 说明 |
|:---|:---|:---|
| 混合（默认） | `['mamba','mamba','attention','attention']` | 前层局部 SSM 建模 + 后层全局语义聚合 |
| 纯 Mamba | `[]` 或 `null` | 全部使用 BiMambaBlock 层 |

主要训练超参：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `encoder_layer_types` | `['mamba','mamba','attention','attention']` | 逐层类型：`mamba` 或 `attention` |
| `backbone_type` | `lynxnet` | 扩散主干：`lynxnet` / `wavenet` / `mamba` |
| `lr_scheduler_args` | CosineAnnealingLR(T_max=160k, eta_min=1e-5) | SSM 在 StepLR 下 loss 容易尖峰 |
| `backbone_args.dropout_rate` | 0.1 | 主干 dropout |
| `optimizer_args.weight_decay` | 1e-5 | L2 权重衰减 |
| `accumulate_grad_batches` | 2 | 梯度累积，平滑噪声 |

声学模型默认 `backbone_type: lynxnet`；要全 Mamba 主干改成 `mamba`（推理更慢，需重训）。

## TuneLab 引擎扩展

完整文档：`extensions/ChoristaDiffsinger-SSM/README.md`（如缺失看 `pack.ps1` 注释）。

构建并部署：

```powershell
cd extensions/ChoristaDiffsinger-SSM
./pack.ps1 -Configuration Release
Copy-Item .\dist\Diffsinger-SSM.tlx 'D:\TuneLab\Diffsinger-SSM.tlx'
```

拖进 TuneLab 安装重启即可。Part 属性面板里能看到这些 SSM 专属选项：

| 选项 | 默认 | 作用 |
|---|---|---|
| `RenderSteps` | 20 | reflow euler 步数；20 步与 5 步质量接近，10 步反而最差 |
| `RenderDepth` | 1.0 | 浅扩散源比，仅在模型训练带 `use_shallow_diffusion` 时有效 |
| `TensorCache` | true | 整句渲染缓存开关 |
| `RenderDevice` | Auto | Auto/CPU/GPU；切换会先 `FreeMemory()` 再用新 EP 重建 session |
| `Phonemizer` | Default | 覆盖 character.yaml 的默认音素器 |
| `PitchTransitionTime` | 0.12 | 相邻音符 pitch 过渡时间 (s) |
| `MinSegmentSpacingMs` | 0 | 短于该值的相邻段会合并为一次合成 |

## 当前推理性能（CPU，Reflow euler，aco_testssm2）

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| linguistic | 103 ms | 1.1 % |
| dur | 3 ms | <1 % |
| pitch | 207 ms | 2.2 % |
| variance | 141 ms | 1.5 % |
| **acoustic (LYNXNet × 20 step)** | **7733 ms** | **83.1 %** |
| vocoder | 1121 ms | 12.0 % |
| 合计 | 9307 ms | RTF ≈ 3.11 |

`tools/sweep_acoustic_steps.py` 单独跑 acoustic：20→2738 ms，10→1448 ms (1.89×)，5→830 ms (3.30×)，2→259 ms (10.57×)。

## 已知坑 / 设计决定

- ONNX 内部 `RandomNormal` 不接受外部 seed，每次 Run 噪声不同，跨 run 比对 SNR 是噪声不是模型差异
- aco onnx 把 `steps` 暴露为动态标量输入 → 改步数不需重导
- `DiffsingerPreferences.Default` 是 mutable static，Render 前直接覆盖即可；引擎扩展就是这么做的
- 渲染缓存路径：`%USERPROFILE%\.TuneLab\ChoristaDS-SSM\RenderCache\`，调参验证时先清掉避免 hash 命中旧结果
- `IVoiceEngine` / `IVoiceSource` API 不能访问轨道列表 / 音频图 / UI 树；任何 "在 TuneLab 里挂 VST"、"做混合轨"、"通过 ARA 嵌入 DAW" 都要 fork TuneLab 主程序，tlx 这层做不到

## 参考

- 论文：[DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
- 上游：[openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)
- Mamba：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) / `mamba-ssm >= 2.3.0`
- 声码器：[NSF](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) + [HiFi-GAN](https://github.com/jik876/hifi-gan)
- 基频：[RMVPE](https://github.com/Dream-High/RMVPE)

## 免责声明

禁止使用本仓库任何功能在未经他人同意的情况下生成其语音，包括但不限于政府领导人、政治人物、名人。违者可能违反版权及人格权相关法律。

## 许可

[Apache 2.0](LICENSE)