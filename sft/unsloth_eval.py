from unsloth import FastLanguageModel
import torch

from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

import argparse
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.multi_label_bin import process_mult


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


alpaca_prompt = """You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def collate_fn(batch):
    # Each `batch[i]` is a dictionary with a "messages" field
    refs = [sample["messages"][3]["content"] for sample in batch]
    prompts = [
        alpaca_prompt.format(
            sample["messages"][1]["content"], sample["messages"][2]["content"], ""
        )
        for sample in batch
    ]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
    )
    tokenized["ref"] = refs
    return tokenized


similarity_scores = []
resps = []
refs = []
generated_responses = []
references = []

dataset = load_from_disk(args.dataset_path)
max_seq_length = 8192
checkpoint = args.model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
model.eval()

loader = DataLoader(dataset["valid"], batch_size=5, collate_fn=collate_fn)

for batch in loader:
    ref_batch = batch.pop("ref")
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.no_grad():
        generated = model.generate(
            **batch,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off prompt inputs to get only the generated completion
    generated_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(batch["input_ids"], generated)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    generated_responses += responses
    references += ref_batch

process_mult(references, generated_responses)
