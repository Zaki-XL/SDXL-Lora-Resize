# SDXL Model & LoRA Quantizer & Rank Resizer

[日本語](README.md) | [English](README_EN.md) | **简体中文**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)

**SDXL Model & LoRA Quantizer & Rank Resizer** 是一款专为 Stable Diffusion XL (SDXL) 及现代生成式 AI 模型（Checkpoints / LoRA）设计的桌面端 GUI 工具。通过将 **SVD (奇异值分解) 降低 Rank** 与 **高效量化 (FP8 / INT8 / INT4 / NVFP4 / GGUF)** 相结合，实现模型文件的极限轻量化与加速。

<p align="center">
  <img src="assets/scr.png" alt="SDXL Model & LoRA Quantizer UI" width="95%">
</p>

---

## 🌟 核心特性

### 1. SVD Rank 尺寸调整 (奇异值分解)
- 对高 Rank LoRA（Rank 128 / 64 等）的多余参数进行截断压缩，在保留画风与表现力的前提下大幅削减体积。
- **Alpha 等比例缩放校正**:
  $$\text{新 Alpha} = \text{原 Alpha} \times \frac{\text{目标 Rank}}{\text{原 Rank}}$$
  在调整 Rank 时 100% 保持实际应用强度比（$\alpha / \text{rank}$），彻底解决 WebUI 或 ComfyUI 加载时画面全黑（Black Screen）或画质崩溃的问题。
- 自动继承并更新训练元数据（`ss_network_dim`, `ss_network_alpha` 等）。

### 2. 丰富的量化精度支持
根据您的显卡世代和使用环境（WebUI / Forge / ComfyUI）选择最合适的格式：

| 格式 | 体积缩减 | 推荐显卡 | 特点与兼容性 |
| :--- | :---: | :--- | :--- |
| **FP8 (e4m3fn)** | 约 50% | RTX 40xx / 50xx | **【首选・高品质】** 画质损耗极低，Forge / ComfyUI 原生支持 |
| **FP8 (e5m2)** | 约 50% | RTX 40xx / 50xx | 宽动态范围 8 位浮点，有效防止数值溢出 |
| **INT8** | 约 50% | 全代 RTX (20xx-50xx) | 通用 8 位整数量化，广泛兼容 Turing、Ampere、Ada 架构 |
| **INT4 / NF4** | 约 75% | 低显存环境 | 极限降低显存占用的 4 位量化（支持 ComfyUI-NF4） |
| **NVFP4 / FP4** | 约 75% | RTX 50xx (Blackwell) | 面向次世代 Blackwell 架构的原生 4 位浮点硬件加速 |
| **FP16** | 约 50% * | 所有 GPU / CPU | 全平台完全兼容的标准 16 位浮点（适合压缩 FP32 原始模型） |
| **GGUF (Q4_K_M)** | 约 70% | 所有 GPU / CPU | 可被 ComfyUI-GGUF 节点及 sd.cpp 直接读取的高效格式 |
| **保持原精度** | - | 所有环境 | 不进行量化，仅执行 SVD Rank 缩减（保持 BF16/FP16） |

\* *对比 FP32 原始文件的缩减比例。*  
\* **注意**: 作者手头没有 RTX 50 系列显卡，因此未在 RTX 50xx 真机上进行过测试（非常欢迎社区的反馈与测试报告！）。

### 3. VAE 自动保护机制
- 自动识别 Checkpoint 模型内部的 VAE 张量（`first_stage_model` 等），严格保持 FP16/FP32 精度，杜绝潜在空间解码失真与黑图。

### 4. 低显存流式转换 (Streaming)
- 超过 6GB 的大型模型无需全部载入显存，逐个张量在 GPU/CPU 之间流式计算与回收。
- **即使在 6GB〜8GB 显存的入门级显卡上，也能安全批量转换，彻底避免 CUDA Out of Memory (OOM) 显存溢出。**

### 5. 多语言无缝切换 (i18n: 日本語 / English / 简体中文)
- 启动时动态扫描 `lang/` 目录。
- 界面右上角支持一键实时切换语言，无需重启程序。
- 仅需在 `lang/` 文件夹放入新的 JSON（如 `ko.json`），即可免编译扩展新语言。

### 6. 关联预览图与元数据自动同步
- 自动复制并重命名同目录下的模型预览图（`.png`, `.jpg`, `.webp`）以匹配转换后的新文件名。
- **[ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) 深度支持**:
  全面兼容流行扩展 ComfyUI-Lora-Manager，自动修改并同步 `*.metadata.json` 内部的模型文件名、新 Rank、新 Alpha、量化标签及转换历史记录。

### 7. 损坏文件预检与完整性校验
- 拖放文件时自动预检 Safetensors 文件头，对损坏或 0 字节文件进行红字加粗高亮，并自动从转换列表中剔除。
- 转换完成后自动校验键名、Shape 形状并检测 NaN/Inf 异常值。若检测到异常，将自动物理删除损坏文件以确保安全。

---

## 🖥️ 运行环境要求

- **操作系统**: Windows 10 / 11 (64-bit)
- **显卡 (GPU)**: NVIDIA GeForce RTX 20xx / 30xx / 40xx / 50xx 系列（支持 CUDA 11.8 或 12.x 驱动）  
  *(※ 作者本人没有 RTX 50 系列显卡，因此未在 50xx 上进行过测试)*
- **Python**: 3.10 或 3.11
- **依赖库**: PyTorch, PyQt6, safetensors（通过安装脚本自动配置）

---

## 🚀 安装与启动

### 第一步: 克隆代码仓库
```bash
git clone https://github.com/Zaki-XL/SDXL-Lora-Resize.git
cd SDXL-Lora-Resize
```

### 第二步: 配置运行环境 (仅首次需要)
双击运行仓库根目录下的 `setup_env.bat`。
该脚本将自动创建 Python 虚拟环境 (`.venv`)，安装 CUDA 版本的 PyTorch 及依赖项，并自动编译启动器 (`SDXL_Quantizer.exe`)。

```bash
setup_env.bat
```

### 第三步: 启动程序
您可以通过以下任意方式启动：

- 双击 **`SDXL_Quantizer.exe`**（带防重复启动保护的启动器）
- 或双击 **`run.bat`**

> [!NOTE]
> 配置文件 `config.ini` 会在首次启动时自动生成。它已被 Git 规则严格忽略，以确保首次运行时的干净配置状态。

---

## 📖 使用指南

```text
[1. 拖入文件] ──> [2. 设置 SVD Rank] ──> [3. 设置量化精度] ──> [4. 点击开始转换]
```

1. **添加文件**:
   - 将需要转换的 `.safetensors` 文件（LoRA 或 Checkpoint）拖入上方虚线框中（或点击“＋ 选择文件...”）。
2. **设置 SVD Rank 缩减**:
   - 选择目标 Rank（如 `Target Rank 32`、`Target Rank 16`、`不调整Rank` 等）。
3. **设置量化精度**:
   - 选择量化格式（如 `FP8 (e4m3fn)`、`INT8`、`INT4/NF4` 等）。
   - 根据需求勾选“保护 VAE (保持FP16)”和“同名文件强制覆盖”。
4. **选择保存目录**:
   - 默认输出至原始文件所在目录。可点击“更改...”选择自定义输出目录。
5. **开始处理**:
   - 点击 **“🚀 开始调整尺寸与量化”**。
   - 在进度条与日志窗口中实时查看转换速度（MB/s）与执行状态。

---

## 📁 项目目录结构

```text
SDXL-Lora-Resize/
├── .gitignore               # Git 忽略规则 (.venv, config.ini, 模型权重等)
├── assets/                  # 图标及图片资源
├── lang/                    # 多语言本地化字典 (JSON)
│   ├── ja.json              # 日语 (日本語)
│   ├── en.json              # 英语 (English)
│   └── zh.json              # 简体中文
├── src/
│   ├── main.py              # 程序入口
│   ├── gui/                 # PyQt6 界面实现与后台工作线程
│   ├── quantizer/           # SVD 缩减与量化核心计算流
│   └── utils/               # i18n, GPU 检测, Safetensors 读写, 配置管理
├── tests/                   # 自动化单元测试套件
├── Launcher.cs              # C# 编写的防重复启动器源码
├── build_launcher.bat       # 启动器独立构建脚本
├── setup_env.bat            # 虚拟环境配置与启动器自动编译脚本
├── run.bat                  # 快捷启动脚本
├── requirements.txt         # 依赖项清单
├── LICENSE                  # MIT 开源许可证
├── README.md                # 日语文档 (日本語)
├── README_EN.md             # 英语文档 (English)
└── README_ZH.md             # 简体中文文档 (本文档)
```

---

## 🧪 运行测试

如需运行自动化单元测试套件，请执行以下命令：

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 📄 开源许可证

本项目源码基于 [MIT License](LICENSE) 开源。  
※ 本工具依赖的 GUI 库 PyQt6 遵循 GNU GPL v3 协议。

---

## 🤝 免责声明

- 使用及分发由本工具转换生成的模型文件时，请严格遵守原始模型的开源许可协议（如 CreativeML Open RAIL++-M、OpenRAIL-M 等）。
- 进行批量模型转换前，建议对重要原始模型进行备份。
