#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""
set -ex

chunk_len=96  # you can change to 32，64 or 96 

dir=./k2fsa-zipformer-bilingual-zh-en-t
mkdir -p $dir/exp_onnx_$chunk_len
dir_onnx=$dir/exp_onnx_$chunk_len

./utils/export-onnx-zh.py \
  --tokens $dir/data/lang_char_bpe/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $dir_onnx \
  --pt-model $dir/epoch-99.pt \
  --decode-chunk-len $chunk_len \
  \
  --num-encoder-layers 2,2,2,2,2 \
  --feedforward-dims 768,768,768,768,768 \
  --nhead 4,4,4,4,4 \
  --encoder-dims 256,256,256,256,256 \
  --attention-dims 192,192,192,192,192 \
  --encoder-unmasked-dims 192,192,192,192,192 \
  \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512

python $dir/cut_reciprocal_encoder.py \
  --onnx_ori $dir_onnx/encoder-epoch-99-avg-1.sim.onnx \
  --onnx_new $dir_onnx/encoder-epoch-99-avg-1.sim.onnx