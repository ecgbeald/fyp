from unsloth import FastLanguageModel
import torch

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import datetime
from torch.utils.data import DataLoader

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss
from sentence_transformers import SentenceTransformer, util
import ast
import re

def parse_label_string(s): # parsing labels such as [1,2,3], [4]
    s = s.strip()
    if not s:
        return [0]
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            return [int(x) for x in parsed]
        else:
            return [0]
    except Exception:
        return [0]

def multi_label(response):
    matched = False
    for line in response.split("\n"):
        if "reason" in line:
            matched = True
            match = re.search(r'"reason":\s*"([^"]*)"', line)
            if match:
                reason_str = match.group(1)
                return parse_label_string(reason_str)
            else:
                return [0]
    if not matched:
        return [0]
        
def parse_explain(response):
    matched = False
    for line in response.split("\n"):
        if "explanation" in line:
            matched = True
            match = re.search(r'"explanation":\s*"([^}]*)"', line)
            if match:
                reason_str = match.group(1)
                return reason_str
            else:
                return ""
    if not matched:
        return ""

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
        alpaca_prompt.format(sample["messages"][1]["content"], sample["messages"][2]["content"], "")
        for sample in batch
    ]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side='left',
        truncation=True,
    )
    tokenized["ref"] = refs
    return tokenized

similarity_scores = []
resps = []
refs = []
generated_responses = []
references = []

dataset = load_from_disk("../data/dataset.hf")
max_seq_length = 8192
checkpoint='/rds/general/user/rm521/home/fyp/step3-sft/Qwen2.5-0.5B-Merged_0205_13-57'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False
)
FastLanguageModel.for_inference(model)
model.eval()

loader = DataLoader(dataset['valid'], batch_size=5, collate_fn=collate_fn)

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
        output[len(input_ids):]
        for input_ids, output in zip(batch["input_ids"], generated)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    generated_responses += responses
    references += ref_batch

st_model = SentenceTransformer('all-MiniLM-L6-v2')
for ref, pred in zip(references, generated_responses):
    ref_label = multi_label(ref)
    pred_label = multi_label(pred)
    refs.append(ref_label)
    resps.append(pred_label)
    if (ref_label == [0] or pred_label == [0]):
        continue
    ref_exp = parse_explain(ref)
    pred_exp = parse_explain(pred)
    ref_emb = st_model.encode(ref_exp, convert_to_tensor=True)
    pred_emb = st_model.encode(pred_exp, convert_to_tensor=True)
    sim_score = util.cos_sim(ref_emb, pred_emb).item()
    similarity_scores.append(sim_score)
    print(ref_exp)
    print(pred_exp)
    print("\n")

mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(refs)
y_pred = mlb.transform(resps)
class_labels = [str(label) for label in mlb.classes_]
print(class_labels)
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
hamming = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hamming}")

average_score = np.mean(similarity_scores)
print(f"Similarity Score for Explanation:{average_score}")