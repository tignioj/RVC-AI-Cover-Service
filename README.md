# RVC AI Cover Service

这是一个面向 [AstrBot AI 翻唱插件](https://github.com/tignioj/astrbot_plugin_ai_cover) 的 RVC 后端服务。它在 RVC WebUI 的基础上增加了 FastAPI 接口，依次完成人声/伴奏分离、去混响、RVC 音色转换和 MP3 混音。

主要特性：

- 提供 `/health`、`/models`、`/cover` 和 `/cache` 接口；
- 通过 PyMSS/MSST 完成人声分离和去混响；
- 自动匹配 RVC `.pth` 模型和对应的 `.index`；
- 按原始音频 SHA-256 缓存干人声与伴奏，重复翻唱可跳过分离步骤；
- GPU 任务串行执行，避免多个模型同时争抢显存；
- 支持共享令牌、上传大小和音频时长限制。

> 本仓库基于 [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 开发，保留其 MIT 许可证及相关第三方许可证。模型权重、索引、FFmpeg、运行时和生成音频不包含在 Git 仓库中。

## 快速开始

### 1. 准备模型与运行环境

按照后文的[环境配置](#环境配置)和[模型与运行目录](#模型与运行目录)安装依赖并下载基础模型。AI 翻唱至少需要：

```text
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pymss_weights/              # 去伴奏、去混响模型
assets/weights/角色名.pth          # RVC 音色模型
logs/**/角色名相关文件.index        # 可选，RVC 特征索引
```

Windows 整合包还需要在项目根目录提供 `ffmpeg.exe` 和 `ffprobe.exe`。

### 2. 启动 AI 翻唱服务

Windows 整合包可直接运行：

```bat
start-ai-cover-service.bat
```

也可以手动启动：

```powershell
runtime\python.exe tools\ai_cover_service.py
```

默认监听 `0.0.0.0:18888`。服务暴露到局域网时，建议先设置一个随机共享令牌，并在防火墙中只允许 AstrBot 所在主机访问：

```powershell
$env:AI_COVER_API_TOKEN = "请替换为随机长字符串"
runtime\python.exe tools\ai_cover_service.py
```

可用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AI_COVER_HOST` | `0.0.0.0` | 监听地址 |
| `AI_COVER_PORT` | `18888` | 监听端口 |
| `AI_COVER_API_TOKEN` | 空 | `X-AI-Cover-Token` 共享令牌；空值表示不鉴权 |
| `AI_COVER_MAX_UPLOAD_MB` | `200` | 最大上传大小，MiB |
| `AI_COVER_MAX_AUDIO_SECONDS` | `900` | 最大音频时长，秒 |
| `AI_COVER_JOB_ROOT` | `TEMP/ai_cover` | 临时任务目录 |
| `AI_COVER_CACHE_ROOT` | `TEMP/ai_cover_cache` | 分离缓存目录 |
| `AI_COVER_LOG_LEVEL` | `INFO` | 日志级别 |

检查服务：

```powershell
$headers = @{ "X-AI-Cover-Token" = "请替换为上面设置的令牌" }
Invoke-RestMethod http://127.0.0.1:18888/health -Headers $headers
Invoke-RestMethod http://127.0.0.1:18888/models -Headers $headers
```

未设置 `AI_COVER_API_TOKEN` 时可以省略 `-Headers $headers`。

## 配合 AstrBot 插件使用

### 1. 安装插件

插件仓库：<https://github.com/tignioj/astrbot_plugin_ai_cover>

在 AstrBot WebUI 的插件管理页面安装该仓库，或者克隆到 AstrBot 的插件目录：

```bash
cd data/plugins
git clone https://github.com/tignioj/astrbot_plugin_ai_cover.git
```

安装后重启 AstrBot，或在插件管理页面重载插件。

### 2. 配置连接

在插件配置页填写：

- `service_url`：AstrBot 能够访问的服务地址，例如 `http://192.168.1.20:18888`；
- `api_token`：与 RVC 主机上的 `AI_COVER_API_TOKEN` 完全一致；
- `timeout_seconds`：单次任务超时，默认 3600 秒；
- 其余参数可保留默认值，再按效果调整。

如果 AstrBot 运行在 Docker 中，`127.0.0.1` 指向的是 AstrBot 容器自身，不能用来访问另一台 Windows 主机；请填写 Windows RVC 主机的局域网 IP。

### 3. 使用命令

翻唱命令需要在同一条消息中附带音频，或回复一条已有音频：

```text
/翻唱模型
/翻唱 珊瑚宫心海
/翻唱 珊瑚宫心海 0
/翻唱 珊瑚宫心海 0 1.3
/翻唱 珊瑚宫心海 0 1.3 0.8
/翻唱状态
```

完整格式：

```text
/翻唱 模型 [升降调] [人声音量] [伴奏音量]
```

| 参数 | 范围 | 默认行为 |
| --- | --- | --- |
| 模型 | `/翻唱模型` 返回的名称 | 必填 |
| 升降调 | `-24`～`24` 的整数 | `0` |
| 人声音量 | `0`～`2` | 使用插件配置，`1` 为原音量 |
| 伴奏音量 | `0`～`2` | 使用插件配置，`1` 为原音量 |

音量为 `0` 表示静音。直接调用后端 API 时音量上限为 `3`，AstrBot 插件为便于日常使用将命令和配置范围限制为 `0`～`2`。

## 分离缓存

服务使用上传文件原始字节的 SHA-256 作为缓存键。相同文件再次翻唱时会复用干人声和伴奏，与模型、升降调和文件名无关。

```text
GET    /cache   查看缓存条目数和占用空间
DELETE /cache   清空已识别的分离缓存
```

AstrBot 插件详情页也提供“分离缓存”管理页面。删除缓存会等待当前 GPU 任务结束，不会删除正在使用的音轨。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务状态、模型数量和缓存统计 |
| `GET` | `/models` | RVC 模型与索引状态 |
| `POST` | `/cover` | 上传音频并生成 MP3 |
| `GET` | `/cache` | 查看分离缓存 |
| `DELETE` | `/cache` | 清空分离缓存 |

`POST /cover` 使用 `multipart/form-data`，必填字段为 `audio` 和 `model`。可选字段为 `transpose`、`index_rate`、`rms_mix_rate`、`protect`、`vocal_gain` 和 `instrumental_gain`。启用令牌后，所有接口都需要请求头 `X-AI-Cover-Token`。

---

## 上游 RVC 使用说明

<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
简单易用的 语音音色转换/变声器 框架<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


[**更新日志**](./docs/cn/Changelog_CN.md) | [**常见问题解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·5毛钱训练AI歌手**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**对照实验记录**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**在线演示**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](./docs/en/README.en.md) | [**中文简体**](./README.md) | [**日本語**](./docs/jp/README.ja.md) | [**한국어**](./docs/kr/README.ko.md) ([**韓國語**](./docs/kr/README.ko.han.md)) | [**Français**](./docs/fr/README.fr.md) | [**Türkçe**](./docs/tr/README.tr.md) | [**Português**](./docs/pt/README.pt.md)

</div>

> 底模使用接近50小时的开源高质量VCTK训练集训练，无版权方面的顾虑，请大家放心使用

> 请期待RVCv3的底模，参数更大，数据更大，效果更好，基本持平的推理速度，需要训练数据量更少。

<table>
   <tr>
		<td align="center">训练推理界面</td>
		<td align="center">实时变声界面</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-webui.bat</td>
		<td align="center">go-realtime_gui.bat</td>
	</tr>
  <tr>
    <td align="center">可以自由选择想要执行的操作。</td>
		<td align="center">我们已经实现端到端170ms延迟。如使用ASIO输入输出设备，已能实现端到端90ms延迟，但非常依赖硬件驱动支持。</td>
	</tr>
</table>

## 简介
本仓库具有以下特点
+ 使用top1检索替换输入源特征为训练集特征来杜绝音色泄漏
+ 即便在相对较差的显卡上也能快速训练
+ 使用少量数据进行训练也能得到较好结果(推荐至少收集10分钟低底噪语音数据)
+ 可以通过模型融合来改变音色(借助ckpt处理选项卡中的ckpt-merge)
+ 简单易用的网页界面
+ 可调用pymss/MSST模型来快速分离人声和伴奏
+ 使用最先进的[人声音高提取算法InterSpeech2023-RMVPE](#参考项目)根绝哑音问题，速度快、资源占用小
+ A卡/I卡使用 CPU 依赖方案；Windows 可使用 DirectML，Linux 使用 CPU

点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

## 环境配置

本分支面向 **Python 3.12 x64**，请先进入仓库根目录。Ubuntu 推荐使用 Ubuntu 24.04 x86_64。

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

安装 Python 3.12 x64 后创建虚拟环境：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 按硬件选择依赖

| 硬件 | 安装方式 |
| --- | --- |
| CPU、AMD、Intel | 使用 `requirments_cpu_py312.txt`；Windows 可使用 DirectML，Linux 使用 CPU |
| NVIDIA RTX 50 系 | 先安装 CUDA 12.8 版 Torch，再安装 `requirments_cu128_py312.txt` |
| NVIDIA RTX 50 系以前 | 先安装 CUDA 11.8 版 Torch，再安装 `requirments_cu118_py312.txt` |

#### CPU、AMD、Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX 50 系：两阶段安装

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### NVIDIA RTX 50 系以前：两阶段安装

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

检查 Torch 与 CUDA 状态：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```


### 修改下载源

三个 `requirments_*.txt` 顶部已经包含下载源。中国大陆用户可保留默认镜像；需要使用官方源时，只替换 `--index-url` 和 `--extra-index-url`，保留包版本、CUDA 后缀和两阶段顺序。

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## 模型与运行目录

WebUI 会自动创建运行目录。模型请从 [Hugging Face 模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 下载，并保持以下路径：

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── pymss_weights/
├── weights/        # user RVC .pth models
└── indices/        # user .index files
logs/
└── mute/           # training silence samples

# Exact paths used by the code
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pretrained/*.pth
assets/pretrained_v2/*.pth
assets/pymss_weights/*
assets/weights/*.pth
assets/indices/*.index
logs/mute/*
```

### 下载模型

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for pymss/MSST vocal separation
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pymss_weights/*" --local-dir assets
```

仅 Windows AMD/Intel DirectML 环境还需要：

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

### FFmpeg

Ubuntu 已在前面的系统依赖命令中安装 FFmpeg。Windows 用户可把下面两个文件放到项目根目录：

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## 开始使用

启动 WebUI：

```bash
python webui.py
```

无桌面的 Ubuntu 服务器：

```bash
python webui.py --noautoopen
```

默认服务监听端口为 `7865`。用户自己的 `.pth` 模型放入 `assets/weights/`，`.index` 文件放入 `assets/indices/`。

## 参考项目
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [pymss-project/pymss](https://github.com/pymss-project/pymss)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).

## 感谢所有贡献者作出的努力
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
