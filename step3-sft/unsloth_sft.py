from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import datetime
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default="../data/dataset.hf",
    help="Path to the output dataset.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the model.",
)
args = parser.parse_args()


max_seq_length = 2048
lora_rank = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint = args.model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules="all-linear",
    lora_alpha=lora_rank,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)

alpaca_prompt = """You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    messages = examples["messages"]
    instructions = []
    inputs = []
    outputs = []
    for message in messages:
        instructions.append(message[1]["content"])
        inputs.append(message[2]["content"])
        outputs.append(message[3]["content"])
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset = load_from_disk(args.dataset_path)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

now = datetime.datetime.now()
timestamp = now.strftime("%d%m_%H-%M")

args = SFTConfig(
    output_dir=f"Qwen2.5-7B-SFT_{timestamp}",
    load_best_model_at_end=True,
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    num_train_epochs=6.0,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    bf16=True,
    max_seq_length=max_seq_length,
    eval_strategy="steps",
    eval_steps=50,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

trainer_stats = trainer.train()
print(trainer_stats)
model.save_pretrained_merged(
    f"Qwen2.5-7B-Merged_{timestamp}",
    tokenizer,
    save_method="merged_16bit",
)
