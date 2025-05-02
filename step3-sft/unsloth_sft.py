from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import datetime

max_seq_length = 8192
checkpoint='/rds/general/user/rm521/home/Qwen2.5-0.5B'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
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
    messages = examples['messages']
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
    return { "text" : texts, }

dataset = load_from_disk("../data/dataset.hf")
dataset = dataset.map(formatting_prompts_func, batched = True,)

now = datetime.datetime.now()
timestamp = now.strftime("%d%m_%H-%M")

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
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = args,
)

trainer_stats = trainer.train()
print(trainer_stats)
model.save_pretrained_merged(f"Qwen2.5-0.5B-Merged_{timestamp}", tokenizer, save_method = "merged_16bit",)
