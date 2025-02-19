from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import logging
import torch


# Set the device (replace 'cuda:0' with the appropriate GPU if you have multiple GPUs)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the device for PyTorch
torch.cuda.set_device(device)

model_path = "/vol/bitbucket/rm521/Qwen-7B"
quant_path = "/vol/bitbucket/rm521/Qwen-7B-AWQ"
quantize_config = BaseQuantizeConfig(
    bits=4, # 4 or 8
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.",
    )
]

max_memory = {0: torch.cuda.get_device_properties(0).total_memory}

model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, device_map="auto", max_memory=max_memory)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

model.quantize(examples)

model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
