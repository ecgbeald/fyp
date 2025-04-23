from transformers import DebertaV2Tokenizer, DataCollatorWithPadding, TrainingArguments, DebertaV2ForSequenceClassification, Trainer, default_data_collator
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

def preprocess_function(examples):
    examples['log'] = [str(text) for text in examples['log']]
    return tokenizer(examples['log'], truncation=True, padding='max_length', max_length=512, return_tensors="pt")

lines = []
with open("../data/log/ssl-access.log-20181009", "r") as f:
    lines.extend(f.readlines())

# Create a DataFrame where each line is a row
df = pd.DataFrame(lines, columns=["log"])

# Optionally strip newline characters
df["log"] = df["log"].str.strip()
dataset = Dataset.from_pandas(df)

model_path = "/rds/general/user/rm521/home/fyp/classifier/2304_12-57/checkpoint-8032"
model = DebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels=10, output_attentions=False, output_hidden_states=False)
model.cuda()
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

tokenized_valid = dataset.map(preprocess_function, batched=True)
dataloader = DataLoader(tokenized_valid, batch_size=260, collate_fn=default_data_collator)

model.eval()
result = []
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)
        predictions = (probs > 0.5).int()
        result.extend(predictions.cpu().tolist())

labels = [[i for i, val in enumerate(row) if val == 1] for row in result]
df['pred'] = labels
filtered = df[df['pred'].apply(lambda x: x != [0])]
filtered.to_csv("acunetix.csv", index=False)