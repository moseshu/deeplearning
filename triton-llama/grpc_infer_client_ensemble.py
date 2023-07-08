#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import json

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import time 
from typing import List
import json
def jload(data_path:str)-> List:
    with open(data_path,'r') as f:
        data = json.load(f)
    return data

def jwrite(data_path,data:list):
    with open(data_path,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

        
def load_txt(data_path:str) -> str:
    with open(data_path,'r') as f:
        data = f.read().splitlines()
    return data
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')
    parser.add_argument(
        '-r',
        '--root-certificates',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded root certificates. Default is None.')
    parser.add_argument(
        '-p',
        '--private-key',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded private key. Default is None.')
    parser.add_argument(
        '-x',
        '--certificate-chain',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded certicate chain. Default is None.')
    parser.add_argument(
        '-C',
        '--grpc-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request to server. Default is None.'
    )


    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "ensemble"
#     model_name="gptq_model"

    # Infer
    data_result = []
    count = 0
    samples = jload("./sample.json")

    total_time = 0
    for item in samples:
        inputs = []
        outputs = []
        item['instruction'] = item['instruction'] + post
        text = item['instruction'] 
        
        inputs.append(grpcclient.InferInput('instruction', [1, 1], "BYTES"))

        # Initialize the data
        print(f"inputs:{inputs[0]}")
        inputs[0].set_data_from_numpy(np.array([[text.encode()]], dtype=np.object_))

        outputs.append(grpcclient.InferRequestedOutput('OUTPUT_0'))
        start = time.time()#Initial time
        # Test with outputs
        try:
            results = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=FLAGS.client_timeout,
                headers={'test': '1'},
                compression_algorithm=FLAGS.grpc_compression_algorithm)
            
            #print(f"cost time1111:{(time.time() - start) * 1000}ms")
            
        except InferenceServerException as err:
            msg = err.message()

            if "TritonModelException" in msg:
                msg_start = msg.index("TritonModelException") + 22
                msg_end = msg.index("\nAt:")
                msg = msg[msg_start:msg_end].strip()
                msg_json = json.loads(msg)
                print(msg_json)
            else:
                print(msg)

    #         exit()

    #    statistics = triton_client.get_inference_statistics(model_name=model_name)
    #    print(statistics)
    #    if len(statistics.model_stats) != 1:
    #        print("FAILED: Inference Statistics")
    #        sys.exit(1)

        # Get the output arrays from the results
    #     print(f"results:{results}")
        output0_data = results.as_numpy('OUTPUT_0')

        
        
        print(output0_data[0][0].decode())
        item['triton_result'] = output0_data[0][0].decode()
        end = time.time()
        diff = (end - start) * 1000
        total_time += diff
        print(f"cost time:{diff}ms")
        print(item)
        data_result.append(item)
        count += 1
        print(f"line num:{count}")
#         print(type(output0_data[0][0]))
        print('PASS: infer')
    jwrite("result.json",data_result)
    print(f"total:{count},total_time:{total_time}")
    
    
