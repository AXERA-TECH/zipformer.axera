# Zipformer model convert

本项目包含 zipformer 原始模型导出 onnx、转换成 axmodel 的完整代码，以及板端 Python / C++ 推理示例。

- [x] 模型转换（导出 onnx → 导出结果验证 → 量化数据生成 → pulsar2 量化）
- [x] Python 推理示例（`python/ax_pretrained_infer.py`）
- [x] C++ 推理示例（`cpp/`）

## 目录结构

```
.
|-- README.md
|-- config/                    # pulsar2 量化配置（encoder/decoder/joiner）
|-- export-for-onnx.sh         # 导出 onnx 并验证（对分）
|-- save_inputs_data.sh        # 生成量化校准数据
|-- icefall/  utils/           # 模型代码与导出脚本
|-- python/
|   `-- ax_pretrained_infer.py # 板端 Python 推理（axengine）
`-- cpp/                       # 板端 C++ 推理源码（见 cpp/README.md）
```

转换好的 axmodel 与推理可执行文件发布在
[HuggingFace: AXERA-TECH/Zipformer.axera](https://huggingface.co/AXERA-TECH/Zipformer.axera)。

## 主要功能

## 快速开始  

### 1. 环境配置  
已验证环境：python3.10
```bash
# 创建虚拟环境并激活
conda create -n zipformer python=3.10
conda activate zipformer

# 工程下载  
git clone https://github.com/AXERA-TECH/zipformer.axera.git

# 安装项目依赖  
   1、先安装torch及torchaudio，注意torch版本不能超过2.8.0，目前kaldifeat最高支持到2.8.0
      eg: pip install torch==2.8.0 torchaudio==2.8.0
      安装完执行 pip list 查看 nvidia-cuda-runtime-cu* 版本，因为安装 k2 和 kaldifeat 需要指定 cuda 版本

   2、安装K2库:从 https://k2-fsa.github.io/k2/cuda.html 下载库文件，然后执行安装指令：
      eg: pip install k2-1.24.4.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

   3、安装kaldifeat库：从 https://csukuangfj.github.io/kaldifeat/cuda.html 下载库文件，然后执行安装指令：
      eg: pip install kaldifeat-1.25.5.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
   注意：选择k2和kaldifeat两个库时torch及python版本要匹配

   4、cd Zipformer.axera
      pip install -r requirements.txt

   5、export PYTHONPATH=$PWD:$PYTHONPATH

# 模型下载
链接: https://github.com/AXERA-TECH/zipformer.axera/releases/download/v1.0/epoch-99.pt
下载后将模型放到 k2fsa-zipformer-bilingual-zh-en-t 文件夹下
```

### 2. 模型转换
#### 导出 onnx 模型  
```bash
sh export-for-onnx.sh

chunk_len 参考值为(32,64,96),根据需要调整
```
转换好的onnx: https://github.com/AXERA-TECH/zipformer.axera/releases/tag/v1.0

#### 量化数据生成
```
sh save_inpus_data.sh
```

#### onnx转换成axmodel
```
decoder转换：
pulsar2 build --config config/zipformer_decoder.json

encoder转换：
pulsar2 build --config config/zipformer_encoder.json

joiner转换：
pulsar2 build --config config/zipformer_joiner.json
```

### 3. 板端推理
#### Python
```
python python/ax_pretrained_infer.py \
  --encoder-model-filename <encoder.axmodel> \
  --decoder-model-filename <decoder.axmodel> \
  --joiner-model-filename  <joiner.axmodel>  \
  --tokens inputs/lang_char_bpe/tokens.txt \
  --sound-dir <wav目录>
```

#### C++
```
# 工具链下载与编译见 cpp/README.md
bash cpp/build_ax650.sh      # → cpp/bin/zipformer_asr_ax650
bash cpp/build_ax630c.sh     # → cpp/bin/zipformer_asr_ax630c

# 板端运行
./bin/zipformer_asr_ax650 --audio-dir inputs/test_wavs
```

## 参考
- [icefall](https://github.com/k2-fsa/icefall)

## 技术支持
- Github issues
- QQ 群: 139953715




