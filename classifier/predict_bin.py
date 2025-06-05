from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    default_data_collator,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser(
    description="Unsloth Evaluation (Faster and can actually fit in 1 GPU)."
)
parser.add_argument(
    "--logfile",
    type=str,
    required=True,
    help="Path to the logfile.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model path for the fine-tuned classifier",
)
parser.add_argument(
    "--savename",
    type=str,
    default="classified_logs",
    help="save file name for the classified logs",
)
args = parser.parse_args()


def preprocess_function(examples):
    examples["log"] = [str(text) for text in examples["log"]]
    return tokenizer(
        examples["log"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


# load logs
lines = []
with open(args.logfile, "r") as f:
    lines.extend(f.readlines())

# Create a DataFrame where each line is a row
df = pd.DataFrame(lines, columns=["log"])

# Optionally strip newline characters
df["log"] = df["log"].str.strip()
dataset = Dataset.from_pandas(df)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model_path = args.model
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_path, num_labels=2, output_attentions=False, output_hidden_states=False
)
model.cuda()
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

start_time = time.time()

tokenized_valid = dataset.map(preprocess_function, batched=True)
dataloader = DataLoader(
    tokenized_valid, batch_size=10, collate_fn=default_data_collator
)

model.eval()

result = []
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        result.extend(predictions.cpu().tolist())
end_time = time.time()
max_memory_bytes = torch.cuda.max_memory_allocated()
max_memory_MB = max_memory_bytes / (1024 ** 2)

print(f"Max GPU memory allocated: {max_memory_MB:.2f} MB")

total_time = end_time - start_time
logs_count = len(df)
time_per_log = total_time / logs_count

print(f"Total evaluation time: {total_time:.2f} seconds")
print(f"Average time per log: {time_per_log:.6f} seconds")

df["pred"] = result
filtered = df[df["pred"] == 1]
filtered.to_csv(f"{args.savename}.csv", index=False)
