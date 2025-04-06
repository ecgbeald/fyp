# for submitting script job
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import json

# cured dataset
dataset = load_dataset("json", data_files="/rds/general/user/rm521/home/fyp/data/prompt.json", split='train')
dataset = dataset.train_test_split(test_size=0.2)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

checkpoint='/rds/general/user/rm521/home/fyp/qwen2.5-7B'
max_memory = {0: torch.cuda.get_device_properties(0).total_memory}
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory)
trainer = SFTTrainer(
    model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=SFTConfig(
        output_dir="Qwen2.5-7B-SFT", 
        do_eval=True,
        # reduced to fit in GPU VRAM
        per_device_train_batch_size=2,
    ),
    peft_config=peft_config,
)

trainer.train()