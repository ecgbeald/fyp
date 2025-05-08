from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import tqdm
import argparse
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.multi_label_bin import process_mult


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


def collate_fn(batch):
    # Each `batch[i]` is a dictionary with a "messages" field
    refs = [sample["messages"][2] for sample in batch]
    prompts = [
        tokenizer.apply_chat_template(
            sample["messages"][:2],  # assuming system + user
            tokenize=False,
            add_generation_prompt=True,
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


generated_responses = []
references = []

dataset = load_from_disk(args.dataset_path)
model_name = args.model
max_memory = {0: torch.cuda.get_device_properties(0).total_memory}
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

subset = dataset["valid"]
loader = DataLoader(subset, batch_size=5, collate_fn=collate_fn)

for batch in tqdm.tqdm(loader, desc="Generating Responses"):
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
