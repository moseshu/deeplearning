# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import triton_python_backend_utils as pb_utils
from transformers import LlamaTokenizer

class TritonPythonModel:

    def initialize(self, args):
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])

        # Parse model output configs 
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output_dtype= pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        self.tokenizer = LlamaTokenizer.from_pretrained("preprocessing/1/llama-7b-hf-tokenizer")
        self.tokenizer.pad_token_id = 0

    def execute(self, requests):
        responses = []

        for idx, request in enumerate(requests):
            tokens_batch = pb_utils.get_input_tensor_by_name(request, 'output_ids').as_numpy()
            tokens_len = pb_utils.get_input_tensor_by_name(request, 'sequence_length').as_numpy()
            tokens_cum_prob = pb_utils.get_input_tensor_by_name(request, 'cum_log_probs').as_numpy()
            tokens_out_prob = pb_utils.get_input_tensor_by_name(request, 'output_log_probs').as_numpy()
            #print(tokens_batch, tokens_len, tokens_cum_prob, tokens_out_prob)

            select_idx = np.argmax(tokens_cum_prob)
            select_token = tokens_batch[0][select_idx]
            #print(select_idx, select_token)

            outputs = []
            xs = self.tokenizer.decode(select_token, skip_special_tokens=True)
            #print(xs)
            if "### Response:" in xs:
                xs = xs.split("### Response:")[1].strip()
            else:
                xs = xs
            outputs.append(xs)
            outputs = [outputs]
            #print(outputs)

            output_tensor = pb_utils.Tensor(
                'OUTPUT_0',
                np.array(outputs).astype(self.output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                output_tensor])
            responses.append(inference_response)

        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

