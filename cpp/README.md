# Zipformer C++ 推理示例

基于 AXEngine 的流式 Zipformer ASR C++ 示例，与 Python 示例
（`ax_pretrained_infer.py`）推理逻辑完全对齐：

- 16 kHz 单声道 PCM WAV 输入
- 80 维 fbank 特征（25 ms 帧长 / 10 ms 帧移，dither=0，snip_edges=false）
- 流式分块：segment=103 帧，shift=96 帧（7 帧左上下文重叠）
- encoder → joiner/decoder 贪心搜索（context_size=2，blank_id=0）
- 尾部补 0.3 s 静音

## 目录结构

```
cpp/
|-- CMakeLists.txt
|-- README.md
|-- build_ax630c.sh            # AX630C 交叉编译
|-- build_ax650.sh             # AX650 交叉编译
|-- download_toolchains.sh     # 下载交叉编译器与 BSP SDK
|-- zipformer_asr_ax.cpp       # 主程序
|-- src/
|   |-- engine_wrapper.hpp/cpp # AXEngine 封装
|   `-- wav_reader.hpp         # 16k mono PCM WAV 读取
`-- third_party/               # kaldi-native-fbank + kissfft 源码包
```

## 环境准备

工具链不随 git 提交，编译前先下载（或使用已有的工具链目录）：

```bash
bash cpp/download_toolchains.sh
```

该脚本会下载：

| 组件 | 来源 |
|------|------|
| gcc-arm-9.2-2019.12 (aarch64-none-linux-gnu) | [Arm Developer](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz) |
| ax650n_bsp_sdk | https://github.com/AXERA-TECH/ax650n_bsp_sdk.git |
| ax620e_bsp_sdk（AX630C 用） | https://github.com/AXERA-TECH/ax620e_bsp_sdk.git |

已有工具链时可用环境变量指定路径：

```bash
export TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
export BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out
```

## 编译

```bash
bash cpp/build_ax650.sh     # → cpp/bin/zipformer_asr_ax650
bash cpp/build_ax630c.sh    # → cpp/bin/zipformer_asr_ax630c
```

## 运行

以 HuggingFace 工程（[AXERA-TECH/Zipformer.axera](https://huggingface.co/AXERA-TECH/Zipformer.axera)）
根目录为工作目录，将可执行文件放到 `bin/` 下：

```bash
# 单文件
./bin/zipformer_asr_ax650 --audio inputs/test_wavs/demo.wav

# 整个目录（*.wav）
./bin/zipformer_asr_ax650 --audio-dir inputs/test_wavs
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models-dir` | `inputs/axmodels_650N`（AX630C 版为 `inputs/axmodels_630C`） | encoder/decoder/joiner.axmodel 所在目录 |
| `--tokens` | `inputs/lang_char_bpe/tokens.txt` | BPE 符号表 |
| `--audio` | - | 单个 16 kHz 单声道 PCM WAV |
| `--audio-dir` | `inputs/test_wavs` | 批量识别目录下所有 *.wav |

> 注意：C++ 示例只支持 16 kHz 单声道 PCM WAV；mp3/flac 等其他格式请使用
> Python 示例或先转成 WAV。

## 板端实测（AX650N）

| 测试音频 | 时长 | 处理耗时 | RTF | 识别结果与 Python 版一致 |
|----------|------|----------|-----|--------------------------|
| 0.wav | 10.053 s | 0.388 s | 0.039 | ✓ |
| 2.wav | 4.690 s | 0.195 s | 0.042 | ✓ |
| demo.wav | 4.204 s | 0.159 s | 0.038 | ✓ |

（Python 示例同平台平均 RTF 约 0.173，C++ 示例约 0.039）
