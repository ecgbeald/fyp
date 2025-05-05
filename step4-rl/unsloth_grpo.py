from unsloth import FastLanguageModel
import torch
import re
import os
from vllm import SamplingParams
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer
import datetime
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk


alpaca_prompt = """You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.

### Instruction:
{}

Please respond in the following format, in English:
<thinking>
...(Step-by-step analysis of the log entry.)
</thinking>
<answer>
...(this is where your response goes)
</answer>

### Input:
{}

### Response:
{}"""

def parse_response(text):
    """Extract structured fields from a wrapped response string."""
    pattern = r'''
    ^\{
    \s*"classification"\s*:\s*"(?P<classification>Malicious|Benign)"\s*,\s*
    "reason"\s*:\s*"\[?(?P<reason>[^\]]+)\]?"\s*,\s*
    "explanation"\s*:\s*"(?P<explanation>.*?)"
    \s*\}$
    '''
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)
    if text.startswith("```json"):
        text = text.replace("```json", "").strip("` \n")
    elif text.startswith("```"):
        text = text.strip("` \n")

    match = regex.search(text)
    if not match:
        return None
    return match.groupdict()

def format_reward_func(prompts, completions, reference, **kwargs):
    rewards = []
    for pred, ref in zip(completions, reference):
        pred = pred[0]['content']
        with open("comparison_log123.txt", "a", encoding="utf-8") as f:
            f.write("\n"+"=" * 80 + "\n")
            f.write("[Output]:" + pred)
            pattern = r"^<thinking>(?P<thinking>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>$"
            compiled = re.compile(pattern, re.DOTALL)
            match = compiled.search(pred)
            if not match:
                f.write("Pattern match failed.\n")
                rewards.append(0.0)
                continue
            pattern_dict = match.groupdict()
            thinking = pattern_dict['thinking']
            if thinking.startswith('\n'):
                thinking = thinking[1:]
            if thinking.endswith('\n'):
                thinking = thinking[:-1]
            answer = pattern_dict['answer']
            if answer.startswith('\n'):
                answer = answer[1:]
            if thinking.endswith('\n'):
                answer = answer[:-1]                                                 
            f.write("[Reference]:\n" + ref.strip() + "\n")
            reward = 2
            reward += min(len(thinking) // 5, 3)
            parse_completion = parse_response(answer)
            if parse_completion is None:
                f.write("Parsing failed for completion.\n")
                rewards.append(reward)
                continue
            reward += 3
            parse_reference = parse_response(ref)
            if parse_completion['classification'] == parse_reference['classification']:
                reward += 2
            if parse_completion['reason'] == parse_reference['reason']:
                reward += 5
            rewards.append(reward)
    return rewards

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../step3-sft/Qwen2.5-7B-Instr-Merged_0505_00-01",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 42,
)

grpo_format = '''Please respond in the following format in English:
<thinking>
...(Step-by-step analysis of the log entry.)
</thinking>
<answer>
...(this is where your response goes)
</answer>'''

def map_to_prompt(sample):
    msgs = sample['messages']
    sys = []
    instr = []
    ref = []
    for msg in msgs:
        sys.append(msg[0])
        content = msg[1]['content'] + "\n\n" + grpo_format
        instr.append({'content' : content, 'role': 'user'})
        ref.append(msg[2]['content'])
    texts = []
    for system, instruction in zip(sys, instr):
        # Must add EOS_TOKEN
        texts.append([system, instruction])
    return {"prompt": texts, "reference": ref}

dataset = load_from_disk("../data/dataset.hf")
dataset = dataset.map(map_to_prompt, batched = True,)
dataset["train"] = dataset["train"].remove_columns("messages")
dataset["valid"] = dataset["valid"].remove_columns("messages")
dataset["test"] = dataset["test"].remove_columns("messages")

now = datetime.datetime.now()
timestamp = now.strftime("%d%m_%H-%M")

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    num_generations = 4,
    max_prompt_length = 2048,
    max_completion_length = 512,
    num_train_epochs = 2.0,
    save_steps = 250,
    max_grad_norm = 0.1,
    dataloader_num_workers = 4,
    dataloader_pin_memory = True,
    report_to = "none", # Can use Weights & Biases
    output_dir = f"Qwen2.5-7B-GRPO_{timestamp}",
)
trainer = GRPOTrainer(
    model=model,
    tokenizer = tokenizer,
    processing_class = tokenizer,
    reward_funcs=format_reward_func,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)
trainer.train()
model.save_lora("grpo_saved_lora")
model.save_pretrained_merged("model_lora", tokenizer, save_method = "lora",)
model.save_pretrained_merged("model_aaa", tokenizer, save_method="merged_16bit")
