# for submitting script job
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import datetime

# cured dataset
dataset = load_from_disk("../data/dataset.hf")

checkpoint='/rds/general/user/rm521/home/fyp/qwen2.5-0.5'
max_memory = {0: torch.cuda.get_device_properties(0).total_memory}
now = datetime.datetime.now()
timestamp = now.strftime("%d%m_%H-%M")
max_seq_length = 8192

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir=f"Qwen2.5-0.5B-SFT_{timestamp}",
    load_best_model_at_end=True,
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    num_train_epochs=3.0,
    bf16=True,
    max_seq_length=max_seq_length,
    eval_strategy="steps",
    eval_steps=250,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=args,
    peft_config=peft_config,
)

trainer.train()
trainer.model = trainer.model.merge_and_unload()
trainer.model.save_pretrained(f"Qwen2.5-0.5B-Merged_{timestamp}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.save_pretrained(save_path)