"""Converter for LLaMa checkpoints in the original format from Meta."""

import argparse
import gc
import glob
import os
import json

import ctranslate2
import numpy as np
import sentencepiece as spm
import torch

from ctranslate2.converters.utils import permute_for_sliced_rotary


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--tokenizer_model", required=True, help="Path to the tokenizer model."
    )
    ctranslate2.converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = LlamaConverter(args.model_dir, args.tokenizer_model)
    converter.convert_from_args(args)


class LlamaConverter(ctranslate2.converters.Converter):
    def __init__(self, model_dir, tokenizer_model_path):
        self._model_dir = model_dir
        self._tokenizer_model_path = tokenizer_model_path

    def _load(self):
        sp = spm.SentencePieceProcessor(self._tokenizer_model_path)
        tokens = [sp.id_to_piece(i) for i in range(len(sp))]

        params_path = os.path.join(self._model_dir, "params.json")
        with open(params_path, encoding="utf-8") as params_file:
            params = json.load(params_file)

        spec = ctranslate2.specs.TransformerDecoderModelSpec.from_config(
            params["n_layers"],
            params["n_heads"],
            activation=ctranslate2.specs.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
        )

        spec.register_vocabulary(tokens)
        spec.register_file(self._tokenizer_model_path)

        pattern = os.path.join(self._model_dir, "consolidated.0*.pth")

        for path in sorted(glob.glob(pattern)):
            model = torch.load(path, map_location="cpu")

            self.set_decoder_spec(spec.decoder, model)

            del model

        # Finalize fused self attention input projection.
        for layer_spec in spec.decoder.layer:
            linear_spec = layer_spec.self_attention.linear[0]

            wi = linear_spec.weight
            wi = wi.reshape(wi.shape[0] * wi.shape[1], wi.shape[2])

            wq, wk, wv = np.split(wi, 3)

            wq = permute_for_sliced_rotary(wq, params["n_heads"])
            wk = permute_for_sliced_rotary(wk, params["n_heads"])

            linear_spec.weight = np.concatenate([wq, wk, wv])

        return spec

    def set_decoder_spec(self, spec, model):
        spec.scale_embeddings = False

        spec.layer_norm.gamma = model.pop("norm.weight").clone().numpy()
        spec.embeddings.weight = append(
            spec.embeddings.weight,
            model.pop("tok_embeddings.weight").clone().numpy(),
            axis=1,
        )
        spec.projection.weight = append(
            spec.projection.weight, model.pop("output.weight").clone().numpy(), axis=0
        )

        for i, layer_spec in enumerate(spec.layer):
            self.set_decoder_layer_spec(i, layer_spec, model)
            gc.collect()

    def set_decoder_layer_spec(self, layer, spec, model):
        prefix = "layers.%d" % layer

        spec.self_attention.layer_norm.gamma = (
            model.pop("%s.attention_norm.weight" % prefix).clone().numpy()
        )
        spec.ffn.layer_norm.gamma = (
            model.pop("%s.ffn_norm.weight" % prefix).clone().numpy()
        )

        wq = model.pop("%s.attention.wq.weight" % prefix).clone().numpy()
        wk = model.pop("%s.attention.wk.weight" % prefix).clone().numpy()
        wv = model.pop("%s.attention.wv.weight" % prefix).clone().numpy()
        wo = model.pop("%s.attention.wo.weight" % prefix).clone().numpy()

        spec.self_attention.linear[0].weight = append(
            spec.self_attention.linear[0].weight, np.stack([wq, wk, wv]), axis=1
        )
        spec.self_attention.linear[1].weight = append(
            spec.self_attention.linear[1].weight, wo, axis=1
        )

        w1 = model.pop("%s.feed_forward.w1.weight" % prefix).clone().numpy()
        w2 = model.pop("%s.feed_forward.w2.weight" % prefix).clone().numpy()
        w3 = model.pop("%s.feed_forward.w3.weight" % prefix).clone().numpy()

        spec.ffn.linear_0.weight = append(spec.ffn.linear_0.weight, w1, axis=0)
        spec.ffn.linear_0_noact.weight = append(
            spec.ffn.linear_0_noact.weight, w3, axis=0
        )
        spec.ffn.linear_1.weight = append(spec.ffn.linear_1.weight, w2, axis=1)


def append(current, weight, axis):
    if isinstance(current, np.ndarray):
        weight = np.concatenate([current, weight], axis=axis)
    return weight


if __name__ == "__main__":
    main()
