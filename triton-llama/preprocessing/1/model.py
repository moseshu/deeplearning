# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import triton_python_backend_utils as pb_utils
from transformers import GenerationConfig, LlamaTokenizer

class TritonPythonModel:

    def initialize(self, args):
        self.tokenizer = LlamaTokenizer.from_pretrained("preprocessing/1/llama-7b-hf-tokenizer")
        self.tokenizer.pad_token_id = 0

    def execute(self, requests):
        responses = []

        for idx, request in enumerate(requests):
            instruction = pb_utils.get_input_tensor_by_name(request, 'instruction').as_numpy()
            #print("instruction:", instruction)

            input_ids, attention_mask = self._preprocess(instruction[0][0].decode("utf-8"))
            #print("input_ids:", input_ids)

            input_ids_tensor = pb_utils.Tensor(
                'input_ids',
                input_ids.astype(np.uint32))

            input_lengths_tensor = pb_utils.Tensor(
                'input_lengths',
                np.array([[len(input_ids[0])]], dtype=np.uint32))

            request_output_len_tensor = pb_utils.Tensor(
                'request_output_len',
                np.array([[50]], dtype=np.uint32))

            start_id_tensor = pb_utils.Tensor(
                'start_id',
                np.array([[1]], dtype=np.uint32))

            end_id_tensor = pb_utils.Tensor(
                'end_id',
                np.array([[2]], dtype=np.uint32))

            beam_width_tensor = pb_utils.Tensor(
                'beam_width',
                np.array([[5]], dtype=np.uint32))

            runtime_top_k_tensor = pb_utils.Tensor(
                'runtime_top_k',
                np.array([[10]], dtype=np.uint32))

            runtime_top_p_tensor = pb_utils.Tensor(
                'runtime_top_p',
                np.array([[0.75]], dtype=np.float32))

            repetition_penalty_tensor = pb_utils.Tensor(
                'repetition_penalty',
                np.array([[1.0]], dtype=np.float32))

            temperature_tensor = pb_utils.Tensor(
                'temperature',
                np.array([[0.1]], dtype=np.float32))

            len_penalty_tensor = pb_utils.Tensor(
                'len_penalty',
                np.array([[1.0]], dtype=np.float32))

            random_seed_tensor = pb_utils.Tensor(
                'random_seed',
                np.array([[1]], dtype=np.uint64))

            is_return_log_probs_tensor = pb_utils.Tensor(
                'is_return_log_probs',
                np.array([[True]], dtype=np.bool_))

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_ids_tensor, input_lengths_tensor, request_output_len_tensor, start_id_tensor,
                end_id_tensor, runtime_top_k_tensor, temperature_tensor, len_penalty_tensor, random_seed_tensor, runtime_top_p_tensor, repetition_penalty_tensor, beam_width_tensor, is_return_log_probs_tensor])
            responses.append(inference_response)

        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


    def _preprocess(self, instruction):
        prompt=[]
        prompt.append(self.generate_prompt(instruction))

        #print("prompt", prompt)
        result = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        input_ids = np.array(result["input_ids"])
        attention_mask = np.array(result["attention_mask"])

        return input_ids, attention_mask

    def generate_prompt(self, instruction, input=None):
        PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
        }

        if input:
            PROMPT_DICT['prompt_input'].format_map(instruction=instruction,input=input)

        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
