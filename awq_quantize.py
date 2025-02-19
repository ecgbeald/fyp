# for awq quantisation, not used.
import torch
from awq import AutoAWQForCausalLM

from transformers import AutoTokenizer
import torch

model_path = "/vol/bitbucket/rm521/Qwen-7B"
quant_path = "/vol/bitbucket/rm521/Qwen-7B-AWQ"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")

model.quantize(tokenizer, quant_config=quant_config)
# Save quantized model
model.save_quantized(quant_path, shard_size="4GB")
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')