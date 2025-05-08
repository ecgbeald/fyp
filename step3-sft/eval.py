from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss
from sentence_transformers import SentenceTransformer, util
import ast
import re
import tqdm
import argparse

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


def parse_label_string(s):  # parses "[1,2,3]", "4", "2,3", etc.
    s = s.strip()
    if not s:
        return [0]

    try:
        # Try parsing as a literal (handles "[1,2,3]", "4", etc.)
        parsed = ast.literal_eval(s)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            if not parsed:
                return [0]
            return [int(x) for x in parsed]
    except Exception:
        pass

    # Handle comma-separated values like "2,3"
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []


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


similarity_scores = []
resps = []
refs = []
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

st_model = SentenceTransformer("all-MiniLM-L6-v2")
for ref, pred in zip(references, generated_responses):
    ref_label = multi_label(ref["content"])
    pred_label = multi_label(pred)
    refs.append(ref_label)
    resps.append(pred_label)
    if ref_label == [0] or pred_label == [0]:
        continue
    ref_exp = parse_explain(ref["content"])
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
print(
    "Classification Report:\n",
    classification_report(y_true, y_pred, target_names=class_labels),
)
hamming = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hamming}")

average_score = np.mean(similarity_scores)
print(f"Similarity Score for Explanation:{average_score}")
