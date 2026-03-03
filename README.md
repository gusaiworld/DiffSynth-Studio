# qwen-imagee-dit-demoire

![Python版本](https://img.shields.io/badge/Python-3.8+-blue) ![许可证](https://img.shields.io/badge/License-MIT-orange)

## 📖 项目介绍
本项目基于 **QwenImageEdit** 模型，专注于图像摩尔纹去除任务，提供高效、便捷、高精度的摩尔纹消除解决方案。

摩尔纹是拍摄屏幕、网格状物体时常见的干扰条纹，严重影响图像清晰度和视觉效果。本项目借助QwenImageEdit的图像编辑能力，针对不同场景（屏幕截图、印刷品拍摄、工业图像等）的摩尔纹，实现自动化、智能化去除，同时最大程度保留图像原有细节，无需手动调参，上手门槛极低。

## ✨ 核心特性
- **高效去纹**：基于QwenImageEdit优化算法，快速处理图像，摩尔纹去除效果显著，兼顾速度与精度。
- **细节保留**：去除摩尔纹的同时，最大程度保留图像纹理、色彩、边缘等核心细节，避免图像模糊、失真。
- **多场景适配**：支持屏幕截图、印刷品、织物、网格物体等多种场景下的摩尔纹去除，适配不同分辨率图像。
- **多图支持**：原项目仅支持单张推理，新增支持batch图像的输入，在多图场景下效率更高耗时更少，不仅仅局限于单张图片编辑。
## 📋 环境准备
### 1. 基础环境要求
- Python 3.8 及以上版本
- PyTorch 1.10+（GPU加速推荐，可提升处理速度）
- CUDA 11.3+（可选，GPU加速需配置）

### 2. 依赖安装
克隆项目后，执行以下命令安装依赖：
```bash
# 克隆项目
git clone https://github.com/gusaiworld/DiffSynth-Studio.git


# 安装依赖
cd DiffSynth-Studio
pip install -e .
```

## 🚀 快速开始
本项目支持 **单张图像去摩尔纹** 和 **批量图像去摩尔纹** 两种方式，推荐使用命令行快速体验。

### 方式1：命令行调用（推荐）
#### 单张图像去摩尔纹
```bash
python moire_removal.py --input input_image.jpg --output output_image.jpg
```

#### 批量图像去摩尔纹
```bash
python moire_removal.py --input_dir input_folder --output_dir output_folder
```

#### 参数说明
| 参数 | 说明 | 可选值 | 默认值 |
| :--- | :--- | :--- | :--- |
| --input | 单张输入图像路径（jpg/png格式） | 合法图像路径 | 无 |
| --output | 单张输出图像路径 | 自定义路径 | output.jpg |
| --input_dir | 批量输入图像文件夹路径 | 合法文件夹路径 | 无 |
| --output_dir | 批量输出图像文件夹路径 | 自定义路径 | output_folder |
| --strength | 去摩尔纹强度（值越大，去纹越彻底，可能损失细节） | 0.1~1.0 | 0.7 |

## 📊 效果展示
以下为不同场景下的摩尔纹去除效果对比（左：原图，右：去摩尔纹后）：

### 场景1：屏幕截图摩尔纹
原图：screen_original.jpg | 去纹后：screen_processed.jpg

### 场景2：摩尔纹数据集
| 对比项    | 图片展示                                   |
|--------|----------------------------------------|
| moire图 | ![0282_moire](./assets/0296_moire.jpg) |
| espnet | ![test_0282](./assets/test_0296.jpg)   |
| our    | ![merged_282](./assets/merged_296.jpg) |

> 注：实际效果请参考项目test文件夹中的示例图像。

## 🔧 常见问题
- **问题1：处理速度慢？**  
  解决：确保已安装PyTorch GPU版本，配置CUDA加速；降低图像分辨率或减小批量处理规模。
- **问题2：图像无法正常读取？**  
  解决：检查输入路径是否正确，确保图像格式为jpg/png，避免路径含中文或特殊字符。
- **问题3：依赖安装失败？**  
  解决：请参考DiffSynth-Studio原项目。

## 📌 进阶配置
### 参数微调
通过调整--strength参数，适配不同类型的摩尔纹：
- 弱摩尔纹（轻微干扰）：strength=0.4~0.6
- 中等摩尔纹（明显干扰）：strength=0.6~0.8
- 强摩尔纹（严重干扰）：strength=0.8~1.0

### 自定义模型配置
若需优化特定场景的去纹效果，可修改config.py文件中的模型参数，调整QwenImageEdit的推理配置（如输入分辨率、迭代次数等）。

## 📄 许可证
本项目采用 **MIT许可证** 开源，允许个人和商业使用，修改和分发，需保留原作者声明。

## 🤝 贡献指南
欢迎大家参与项目贡献，具体方式：
1. Fork本仓库
2. 创建新分支（git checkout -b feature/xxx）
3. 提交修改（git commit -m "add xxx feature"）
4. 推送分支（git push origin feature/xxx）
5. 提交Pull Request

贡献方向：优化去纹算法、增加新场景适配、完善API文档、修复已知bug等。

## 📞 联系作者
若有问题、建议或合作需求，可通过以下方式联系：
- GitHub：[gusaiworld](https://github.com/gusaiworld)

---
<center>🌟 感谢使用QwenImageEdit去摩尔纹项目，祝您使用愉快！ 🌟</center>