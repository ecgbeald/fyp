from unsloth import FastLanguageModel
import torch

from datasets import load_from_disk

import re
import tqdm
import argparse
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.multi_label_bin import process_mult

parser = argparse.ArgumentParser(
    description="Unsloth Evaluation (Faster and can actually fit in 1 GPU)."
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="../data/dataset.hf",
    help="Path to huggingface dataset.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model path.",
)
args = parser.parse_args()


def collate_fn(batch):
    # Each `batch[i]` is a dictionary with a "messages" field
    refs = [sample["reference"] for sample in batch]

    prompts = [
        tokenizer.apply_chat_template(
            sample["prompt"][:2],
            tokenize=False,
            add_generation_prompt=False,
        )
        for sample in batch
    ]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    tokenized["ref"] = refs
    return tokenized


similarity_scores = []
resps = []
refs = []
generated_responses = []
references = []

grpo_format = """Please respond in the following format in English:
<thinking>
...(Provide an analysis of the following log entry, justify your reasoning using bullet point)
...(Do NOT analyse the IP address)
</thinking>
<answer>
...(this is where your response goes)
</answer>"""


def map_to_prompt(sample):
    msgs = sample["messages"]
    sys = []
    instr = []
    ref = []
    for msg in msgs:
        sys.append(msg[0])
        content = msg[1]["content"] + "\n\n" + grpo_format
        instr.append({"content": content, "role": "user"})
        ref.append(msg[2]["content"])
    texts = []
    for system, instruction in zip(sys, instr):
        texts.append([system, instruction])
    return {"prompt": texts, "reference": ref}


dataset = load_from_disk(args.dataset_path)
dataset = dataset.map(
    map_to_prompt,
    batched=True,
)
dataset["valid"] = dataset["valid"].remove_columns("messages")

max_seq_length = 2048
checkpoint = args.model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
tokenizer.pad_token = tokenizer.eos_token

for entry in tqdm.tqdm(dataset["valid"], desc="Generating responses"):
    inputs = [
        tokenizer.apply_chat_template(
            entry["prompt"][:2], tokenize=False, add_generation_prompt=True
        )
    ]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
    input_ids = inputs["input_ids"]
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [g[len(i) :] for i, g in zip(input_ids, outputs)]
    decoded = tokenizer.batch_decode(outputs)[0]
    match = re.search(r"^Log Entry:.*$", decoded, re.MULTILINE)
    if match:
        log_entry_line = match.group(0)
        with open("generated_outputs.txt", "a", encoding="utf-8") as f:
            f.write(f"-----------\n")
            f.write(f"Log: {log_entry_line}")
            
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    reference = {"content": entry["reference"]}
    generated_responses.append(response)
    references.append(reference)
    with open("generated_outputs.txt", "a", encoding="utf-8") as f:
        f.write("Reference:\n")
        f.write(reference["content"].strip() + "\n\n")
        f.write("Generated Response:\n")
        f.write(response.strip() + "\n")
        f.write("\n" + "=" * 50 + "\n\n")

generated_answers = []
for pred in generated_responses:
    pattern = (
        r"^<thinking>(?P<thinking>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>$"
    )
    compiled = re.compile(pattern, re.DOTALL)
    match = compiled.search(pred)
    if not match:
        generated_answers.append("")
        continue
    pattern_dict = match.groupdict()
    generated_answers.append(pattern_dict["answer"])

assert len(generated_answers) == len(references)

process_mult(references, generated_answers)
