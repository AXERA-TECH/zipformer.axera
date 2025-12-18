#!/bin/bash
chunk_len=96
repo=./k2fsa-zipformer-bilingual-zh-en-t
dir=$repo/exp_onnx_$chunk_len
output_dir=$repo/exp_onnx_$chunk_len/datasets

./utils/save_inputs_data.py \
  --encoder-model-filename $dir/encoder-epoch-99-avg-1.sim.onnx \
  --decoder-model-filename $dir/decoder-epoch-99-avg-1.sim.onnx \
  --joiner-model-filename $dir/joiner-epoch-99-avg-1.sim.onnx \
  --test-wavs-dir $repo/qua_wavs_100 \
  --output-dir $output_dir \
  --batch-size 1000


