# 环境配置指南

本项目提供三个环境配置文件，适用于不同的使用场景：

## 📋 环境配置文件

### 1. `environment.yml` - Mac M1/M2 专用
```bash
# 快速安装（推荐）
./setup_mac.sh

# 或手动安装
conda env create -f environment.yml
conda activate py312aiwatermark
export KMP_DUPLICATE_LIB_OK=TRUE
```

**适用场景：**
- Mac M1/M2 芯片
- 本地开发和使用
- PyTorch 使用 MPS 加速

**Mac M1 特殊注意事项：**
- 需要设置 `KMP_DUPLICATE_LIB_OK=TRUE` 解决OpenMP冲突
- Flash Attention不支持，使用`--transparent`模式
- 建议使用脚本：`./setup_mac.sh && ./start_mac.sh`

### 2. `environment_cloud.yml` - 云服务器 Python 3.12
```bash
conda env create -f environment_cloud.yml
conda activate py312aiwatermark
export KMP_DUPLICATE_LIB_OK=TRUE
```

**适用场景：**
- 云服务器环境
- Python 3.12
- 支持 CUDA 加速

### 3. `environment_py311.yml` - 云服务器 Python 3.11（稳定版）
```bash
conda env create -f environment_py311.yml
conda activate py311aiwatermark
export KMP_DUPLICATE_LIB_OK=TRUE
```

**适用场景：**
- 云服务器环境
- Python 3.11（更稳定）
- 兼容性最佳

## 🚀 使用方法

### 选择合适的环境配置

1. **Mac 用户**：使用 `environment.yml`
2. **云服务器 + Python 3.12**：使用 `environment_cloud.yml`
3. **云服务器 + 稳定性优先**：使用 `environment_py311.yml`

### 基本使用流程

```bash
# 1. 创建环境（选择对应的配置文件）
conda env create -f environment.yml

# 2. 激活环境
conda activate py312aiwatermark  # 或 py311aiwatermark

# 3. 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 4. 下载模型（可选）
./download_models.sh --recommended  # 下载推荐模型
# 或
python download_florence2.py        # 下载Florence-2模型

# 5. 验证程序
python remwm.py --help

# 6. 使用程序
python remwm.py input.jpg output.png --transparent
```

### 🧭 命令行参数说明（remwm.py）

- 位置参数：
  - input_path：输入文件或目录
  - output_path：输出文件或目录

- 通用选项：
  - --overwrite：批量模式下覆盖已存在文件
  - --transparent：将水印区域变为透明（视频会自动合成到白底）
  - --max-bbox-percent FLOAT：最大检测框占比，默认 10.0
  - --force-format [PNG|WEBP|JPG|MP4|AVI]：强制输出格式（默认图片按输入格式，视频默认 MP4）

- 视频（FFmpeg）相关选项（已将默认值设置为与原行为一致）：
  - --keep-audio/--no-keep-audio：是否合并原视频音频，默认 keep-audio
  - --vcodec CODEC：视频编码器，默认 libx264
  - --crf N：恒定质量因子（x264/x265 有效），默认不指定
  - --bitrate RATE：视频码率（如 5M/2500k）；设置后通常不再使用 CRF，默认不指定
  - --preset NAME：编码预设（ultrafast…veryslow），默认 fast
  - --pix-fmt FMT：像素格式，默认 yuv420p
  - --fps FLOAT：覆盖输出帧率，默认不设置（使用原视频帧率）
  - --threads N：FFmpeg 线程数，默认不设置
  - --start SECONDS：从指定秒开始处理片段，默认不设置
  - --duration SECONDS：处理指定时长的片段，默认不设置

示例：

```bash
# 图片去水印（透明模式）
python remwm.py input.jpg output.png --transparent

# 视频去水印（默认：x264 + preset fast + yuv420p，合并原音频）
python remwm.py input.mp4 out_dir --force-format MP4

# 视频：不保留音频
python remwm.py input.mp4 out_dir --force-format MP4 --no-keep-audio

# 视频：指定码率 5M
python remwm.py input.mp4 out_dir --force-format MP4 --bitrate 5M

# 视频：CRF 20 + veryslow 预设
python remwm.py input.mp4 out_dir --force-format MP4 --crf 20 --preset veryslow

# 视频：只处理第 10 秒开始的 20 秒
python remwm.py input.mp4 out_dir --force-format MP4 --start 10 --duration 20
```


## 🛠️ 故障排除

### 如果环境创建失败

1. **conda镜像源问题**（最常见）：
   ```bash
   # 修复镜像源配置
   conda config --remove-key channels
   conda config --add channels conda-forge
   conda config --add channels defaults
   conda clean --all -y
   ```

2. **手动安装依赖**（如果pip冲突）：
   ```bash
   # 创建基础环境
   conda create -n py312aiwatermark python=3.12 -c conda-forge
   conda activate py312aiwatermark

   # 安装核心依赖
   pip install torch torchvision torchaudio
   pip install transformers diffusers accelerate
   pip install iopaint opencv-python-headless
   pip install tqdm loguru click pillow
   ```

3. **删除现有环境**：
   ```bash
   conda env remove -n py312aiwatermark -y
   # 或
   conda env remove -n py311aiwatermark -y
   ```

4. **尝试其他配置**：
   - 如果 `environment_cloud.yml` 失败，尝试 `environment_py311.yml`
   - 如果仍然失败，使用上面的手动安装方法

5. **检查 conda 版本**：
   ```bash
   conda --version
   conda update conda
   ```

### 常见问题

- **镜像源错误**：清华镜像等第三方源可能不稳定，使用官方源
- **依赖版本冲突**：已移除固定的huggingface_hub版本，让pip自动解决依赖
- **版本冲突**：使用 Python 3.11 配置文件（最稳定）
- **网络问题**：检查代理设置和网络连接
- **权限问题**：确保有写入权限

### 云服务器特殊情况

如果在云服务器上遇到镜像源问题，可以：

1. **修复镜像源**：
   ```bash
   conda config --remove-key channels
   conda config --add channels conda-forge
   conda config --add channels defaults
   ```

2. **禁用插件**：
   ```bash
   conda --no-plugins env create -f environment_cloud.yml
   ```

3. **手动安装**：
   ```bash
   conda create -n py312aiwatermark python=3.12 numpy scipy -c conda-forge
   conda activate py312aiwatermark
   pip install torch torchvision torchaudio transformers diffusers accelerate iopaint opencv-python-headless
   ```

## 📝 环境说明

### 核心依赖版本

| 包名             | Mac 版本 | 云服务器 3.12 | 云服务器 3.11 |
| ---------------- | -------- | ------------- | ------------- |
| Python           | 3.12     | 3.12          | 3.11          |
| PyTorch          | 2.8.0    | 2.8.0         | 2.8.0         |
| NumPy            | 1.26.4   | 1.26.4        | 1.26.4        |
| SciPy            | 1.16.1   | 1.16.1        | 1.16.1        |
| Transformers     | 4.48.3   | 4.48.3        | 4.48.3        |
| Diffusers        | 0.27.2   | 0.27.2        | 0.27.2        |
| IOPaint          | 1.4.3    | 1.4.3         | 1.4.3         |
| Accelerate       | 1.10.0   | 1.10.0        | 1.10.0        |
| Hugging Face Hub | 0.24.6   | 0.24.6        | 0.24.6        |

### 功能特性

- ✅ Florence-2 模型支持（需要 Hugging Face 登录）
- ✅ LaMa 模型支持
- ✅ 图像和视频处理
- ✅ GUI 和命令行界面
- ✅ 透明模式支持

### 🔑 Florence-2 模型配置

Florence-2是Microsoft的AI模型，用于水印检测，需要Hugging Face认证：

1. **获取Token**：
   - 注册 [huggingface.co](https://huggingface.co)
   - 创建Access Token（Read权限）

2. **配置方法**：
   ```bash
   # 方法1：环境变量（推荐）
   export HUGGINGFACE_TOKEN="hf_your_token_here"

   # 方法2：全局登录
   huggingface-cli login

   # 方法3：修改代码
   # 在remwm.py中直接替换token
   ```

3. **下载模型**：
   ```bash
   python download_florence2.py
   ```

**注意**：Mac M1用户如果遇到flash_attn错误，使用`--transparent`模式

### 📹 视频处理需要FFmpeg

项目支持视频水印移除，但需要FFmpeg处理音频：

```bash
# Mac安装
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# 下载FFmpeg并添加到PATH环境变量
# https://ffmpeg.org/download.html
```

**重要**：如果没有安装FFmpeg，视频处理后将没有声音！

### 📦 IOPaint 模型管理

使用专门的模型下载脚本：

```bash
# 下载推荐模型（lama, fcf, mat）
./download_models.sh --recommended

# 下载指定模型
./download_models.sh lama

# 查看所有可用模型
./download_models.sh --list

# 检查已下载的模型
./download_models.sh --check

# 下载所有模型
./download_models.sh --all
```

**可用模型**：
- `lama` - LaMa水印移除模型（推荐）
- `fcf` - 快速卷积填充
- `mat` - 遮罩感知转换器
- `cv2` - OpenCV填充（轻量级）
- `ldm` - 潜在扩散模型
- 等等...

## 💡 推荐使用

- **新用户**：建议使用 `environment_py311.yml`（最稳定）
- **Mac 用户**：使用 `environment.yml`
- **需要最新特性**：使用 `environment_cloud.yml`

环境配置完成后，请参考项目主 README 文件了解具体使用方法。
