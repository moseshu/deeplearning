"""Converter for LLaMa checkpoints in the Hugging Face format."""

import argparse
import gc

import ctranslate2
import numpy as np

from ctranslate2.converters.transformers import register_loader, ModelLoader


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model name or path.")
    ctranslate2.converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = ctranslate2.converters.TransformersConverter(
        args.model,
        copy_files=["tokenizer.model"],
        load_as_float16=True,
        low_cpu_mem_usage=True,
    )
    converter.convert_from_args(args)


@register_loader("LlamaConfig")
class LlamaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "LlamaForCausalLM"

    def get_model_spec(self, model):
        spec = ctranslate2.specs.TransformerDecoderModelSpec.from_config(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            activation=ctranslate2.specs.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
        )

        self.set_decoder(spec.decoder, model.model)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight.numpy()

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            wq = layer.self_attn.q_proj.weight.numpy()
            wk = layer.self_attn.k_proj.weight.numpy()
            wv = layer.self_attn.v_proj.weight.numpy()
            wo = layer.self_attn.o_proj.weight.numpy()

            layer_spec.self_attention.linear[0].weight = np.concatenate([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = wo

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.gate_proj)
            self.set_linear(layer_spec.ffn.linear_0_noact, layer.mlp.up_proj)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.down_proj)

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


if __name__ == "__main__":
    main()
