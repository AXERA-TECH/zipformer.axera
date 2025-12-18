#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)
#
"""
这个脚本用于批量处理test_wavs目录中的所有wav文件，
运行实际推理并保存所有ONNX模型（encoder、decoder、joiner）的真实输入数据到zip文件中。（量化数据）
"""

import argparse
import logging
import os
import glob
import zipfile
import time
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature


def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
    )

    parser.add_argument(
        "--test-wavs-dir",
        type=str,
        required=True,
        help="Directory containing test wav files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets_generate_real",
        help="Output directory for saved npy files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for writing to zip files.",
    )

    return parser


class OnnxModelWithDataSaver:
    def __init__(
        self, 
        encoder_model_filename: str, 
        decoder_model_filename: str,
        joiner_model_filename: str,
        output_dir: str,
        batch_size: int = 100
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        self.session_opts = session_opts
        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)
        self.encoder_counter = 0
        self.decoder_counter = 0
        self.joiner_counter = 0
        self.batch_size = batch_size
        self.pending_files = {}
        
    def _add_to_batch(self, zip_path: str, file_path: str, arcname: str):
        if zip_path not in self.pending_files:
            self.pending_files[zip_path] = []
        self.pending_files[zip_path].append((file_path, arcname))
        if len(self.pending_files[zip_path]) >= self.batch_size:
            self._flush_batch(zip_path)
    
    def _flush_batch(self, zip_path: str = None):
        if zip_path is None:
            zip_paths = list(self.pending_files.keys())
            for zp in zip_paths:
                self._flush_batch(zp)
            return
        if zip_path not in self.pending_files or not self.pending_files[zip_path]:
            return
        with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in self.pending_files[zip_path]:
                zipf.write(file_path, arcname=arcname)
                try:
                    os.remove(file_path)
                except:
                    pass
        self.pending_files[zip_path] = []

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.init_encoder_states()

    def init_encoder_states(self, batch_size: int = 1):
        # decode_chunk_len = 96
        # T = 103
        # num_encoder_layers = "2,2,2,2,2"
        # encoder_dims = "256,256,256,256,256"
        # attention_dims = "192,192,192,192,192"
        # cnn_module_kernels = "31,31,31,31,31"
        # left_context_len = "192,96,48,24,96"

        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        print("encoder_meta:", encoder_meta)

        model_type = encoder_meta["model_type"]
        print("model_type:", model_type)
        assert model_type == "zipformer", model_type

        decode_chunk_len = int(encoder_meta["decode_chunk_len"])
        print("decode_chunk_len:", decode_chunk_len)
        T = int(encoder_meta["T"])
        print("T:", T)

        num_encoder_layers = encoder_meta["num_encoder_layers"]
        print("num_encoder_layers:", num_encoder_layers)
        encoder_dims = encoder_meta["encoder_dims"]
        print("encoder_dims:", encoder_dims)
        attention_dims = encoder_meta["attention_dims"]
        print("attention_dims:", attention_dims)
        cnn_module_kernels = encoder_meta["cnn_module_kernels"]
        print("cnn_module_kernels:", cnn_module_kernels)
        left_context_len = encoder_meta["left_context_len"]
        print("left_context_len:", left_context_len)

        def to_int_list(s):
            return list(map(int, s.split(",")))
        num_encoder_layers = to_int_list(num_encoder_layers)
        encoder_dims = to_int_list(encoder_dims)
        attention_dims = to_int_list(attention_dims)
        cnn_module_kernels = to_int_list(cnn_module_kernels)
        left_context_len = to_int_list(left_context_len)
        num_encoders = len(num_encoder_layers)
        cached_len = []
        cached_avg = []
        cached_key = []
        cached_val = []
        cached_val2 = []
        cached_conv1 = []
        cached_conv2 = []
        N = batch_size
        for i in range(num_encoders):
            cached_len.append(torch.zeros(num_encoder_layers[i], N, dtype=torch.int64))
            cached_avg.append(torch.zeros(num_encoder_layers[i], N, encoder_dims[i]))
            cached_key.append(
                torch.zeros(
                    num_encoder_layers[i], left_context_len[i], N, attention_dims[i]
                )
            )
            cached_val.append(
                torch.zeros(
                    num_encoder_layers[i],
                    left_context_len[i],
                    N,
                    attention_dims[i] // 2,
                )
            )
            cached_val2.append(
                torch.zeros(
                    num_encoder_layers[i],
                    left_context_len[i],
                    N,
                    attention_dims[i] // 2,
                )
            )
            cached_conv1.append(
                torch.zeros(
                    num_encoder_layers[i], N, encoder_dims[i], cnn_module_kernels[i] - 1
                )
            )
            cached_conv2.append(
                torch.zeros(
                    num_encoder_layers[i], N, encoder_dims[i], cnn_module_kernels[i] - 1
                )
            )
        self.cached_len = cached_len
        self.cached_avg = cached_avg
        self.cached_key = cached_key
        self.cached_val = cached_val
        self.cached_val2 = cached_val2
        self.cached_conv1 = cached_conv1
        self.cached_conv2 = cached_conv2
        self.num_encoders = num_encoders
        self.segment = T
        self.offset = decode_chunk_len

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.context_size = 2
        self.vocab_size = 6254

    def init_joiner(self, joiner_model_filename: str):
        self.joiner = ort.InferenceSession(
            joiner_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.joiner_dim = 512

    def save_encoder_inputs(self, x: torch.Tensor):
        timestamp = int(time.time() * 1000000)
        counter = self.encoder_counter
        self.encoder_counter += 1
        x_filename = f"encoder_x_{counter}_{timestamp}.npy"
        x_path = os.path.join(self.output_dir, x_filename)
        np.save(x_path, x.numpy())
        x_zip_path = os.path.join(self.output_dir, "x.zip")
        self._add_to_batch(x_zip_path, x_path, x_filename)
        for name in ["cached_len", "cached_avg", "cached_key", "cached_val",
                     "cached_val2", "cached_conv1", "cached_conv2"]:
            states = getattr(self, name)
            for i, s in enumerate(states):
                fname = f"encoder_{name}_{i}_{counter}_{timestamp}.npy"
                fpath = os.path.join(self.output_dir, fname)
                if isinstance(s, torch.Tensor):
                    np.save(fpath, s.numpy())
                else:
                    np.save(fpath, np.array(s))
                zip_path = os.path.join(self.output_dir, f"{name}_{i}.zip")
                self._add_to_batch(zip_path, fpath, fname)

    def save_decoder_inputs(self, decoder_input: torch.Tensor):
        timestamp = int(time.time() * 1000000)
        counter = self.decoder_counter
        self.decoder_counter += 1
        filename = f"decoder_input_{counter}_{timestamp}.npy"
        fpath = os.path.join(self.output_dir, filename)
        np.save(fpath, decoder_input.numpy())
        zip_path = os.path.join(self.output_dir, "y.zip")
        self._add_to_batch(zip_path, fpath, filename)

    def save_joiner_inputs(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        timestamp = int(time.time() * 1000000)
        counter = self.joiner_counter
        self.joiner_counter += 1
        enc_filename = f"joiner_encoder_out_{counter}_{timestamp}.npy"
        enc_fpath = os.path.join(self.output_dir, enc_filename)
        np.save(enc_fpath, encoder_out.numpy())
        dec_filename = f"joiner_decoder_out_{counter}_{timestamp}.npy"
        dec_fpath = os.path.join(self.output_dir, dec_filename)
        np.save(dec_fpath, decoder_out.numpy())
        enc_zip_path = os.path.join(self.output_dir, "encoder_out.zip")
        self._add_to_batch(enc_zip_path, enc_fpath, enc_filename)
        dec_zip_path = os.path.join(self.output_dir, "decoder_out.zip")
        self._add_to_batch(dec_zip_path, dec_fpath, dec_filename)

    def _build_encoder_input_output(self, x: torch.Tensor):
        encoder_input = {"x": x.numpy()}
        encoder_output = ["encoder_out"]
        def build_states_input(states: List[torch.Tensor], name: str):
            for i, s in enumerate(states):
                if isinstance(s, torch.Tensor):
                    encoder_input[f"{name}_{i}"] = s.numpy()
                else:
                    encoder_input[f"{name}_{i}"] = s
                encoder_output.append(f"new_{name}_{i}")
        build_states_input(self.cached_len, "cached_len")
        build_states_input(self.cached_avg, "cached_avg")
        build_states_input(self.cached_key, "cached_key")
        build_states_input(self.cached_val, "cached_val")
        build_states_input(self.cached_val2, "cached_val2")
        build_states_input(self.cached_conv1, "cached_conv1")
        build_states_input(self.cached_conv2, "cached_conv2")
        return encoder_input, encoder_output

    def _update_states(self, states: List[np.ndarray]):
        num_encoders = self.num_encoders
        self.cached_len = states[num_encoders * 0 : num_encoders * 1]
        self.cached_avg = states[num_encoders * 1 : num_encoders * 2]
        self.cached_key = states[num_encoders * 2 : num_encoders * 3]
        self.cached_val = states[num_encoders * 3 : num_encoders * 4]
        self.cached_val2 = states[num_encoders * 4 : num_encoders * 5]
        self.cached_conv1 = states[num_encoders * 5 : num_encoders * 6]
        self.cached_conv2 = states[num_encoders * 6 : num_encoders * 7]

    def run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        self.save_encoder_inputs(x)
        encoder_input, encoder_output_names = self._build_encoder_input_output(x)
        out = self.encoder.run(encoder_output_names, encoder_input)
        self._update_states(out[1:])
        return torch.from_numpy(out[0])

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        self.save_decoder_inputs(decoder_input)
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]
        return torch.from_numpy(out)

    def run_joiner(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        self.save_joiner_inputs(encoder_out, decoder_out)
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]
        return torch.from_numpy(out)


def create_streaming_feature_extractor() -> OnlineFeature:
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return OnlineFbank(opts)


def read_sound_file(filename: str, expected_sample_rate: float) -> torch.Tensor:
    wave, sample_rate = torchaudio.load(filename)
    # 重采样
    if sample_rate != expected_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=expected_sample_rate
        )
        wave = resampler(wave)
        sample_rate = expected_sample_rate
    assert (
        sample_rate == expected_sample_rate
    ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
    return wave[0].contiguous()


def greedy_search(
    model: OnnxModelWithDataSaver,
    encoder_out: torch.Tensor,
    context_size: int,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
) -> tuple:
    blank_id = 0
    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)
        decoder_out = model.run_decoder(decoder_input)
    else:
        assert hyp is not None, hyp
    encoder_out = encoder_out.squeeze(0)
    T = encoder_out.size(0)
    for t in range(T):
        cur_encoder_out = encoder_out[t : t + 1]
        joiner_out = model.run_joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor([decoder_input], dtype=torch.int64)
            decoder_out = model.run_decoder(decoder_input)
    return hyp, decoder_out


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))
    wav_pattern = os.path.join(args.test_wavs_dir, "*.wav")
    wav_files = glob.glob(wav_pattern)
    wav_files = sorted(wav_files)
    if len(wav_files) == 0:
        logging.error(f"No wav files found in {args.test_wavs_dir}")
        return
    logging.info(f"Found {len(wav_files)} wav files")
    model = OnnxModelWithDataSaver(
        encoder_model_filename=args.encoder_model_filename,
        decoder_model_filename=args.decoder_model_filename,
        joiner_model_filename=args.joiner_model_filename,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    sample_rate = 16000
    blank_id = 0
    context_size = model.context_size
    logging.info("开始处理音频文件...")
    for idx, wav_file in enumerate(wav_files):
        logging.info(f"Processing [{idx+1}/{len(wav_files)}]: {wav_file}")
        online_fbank = create_streaming_feature_extractor()
        waves = read_sound_file(wav_file, sample_rate)
        tail_padding = torch.zeros(int(0.3 * sample_rate), dtype=torch.float32)
        wave_samples = torch.cat([waves, tail_padding])
        num_processed_frames = 0
        segment = model.segment
        offset = model.offset
        hyp = None
        decoder_out = None
        chunk = int(1 * sample_rate)
        start = 0
        while start < wave_samples.numel():
            end = min(start + chunk, wave_samples.numel())
            samples = wave_samples[start:end]
            start += chunk
            online_fbank.accept_waveform(
                sampling_rate=sample_rate,
                waveform=samples,
            )
            while online_fbank.num_frames_ready - num_processed_frames >= segment:
                frames = []
                for i in range(segment):
                    frames.append(online_fbank.get_frame(num_processed_frames + i))
                num_processed_frames += offset
                frames = torch.cat(frames, dim=0)
                frames = frames.unsqueeze(0)
                encoder_out = model.run_encoder(frames)
                hyp, decoder_out = greedy_search(
                    model,
                    encoder_out,
                    context_size,
                    decoder_out,
                    hyp,
                )
        model._flush_batch()
        logging.info(f"Completed: {wav_file}")
    model._flush_batch()
    logging.info(f"全部完成!")
    logging.info(f"  Total encoder chunks: {model.encoder_counter}")
    logging.info(f"  Total decoder chunks: {model.decoder_counter}")
    logging.info(f"  Total joiner chunks: {model.joiner_counter}")
    logging.info(f"数据已保存到: {args.output_dir}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

