from unsloth import FastLanguageModel
import torch

from datasets import load_from_disk
import torch

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss
from sentence_transformers import SentenceTransformer, util
import ast
import re
import tqdm
import argparse

parser = argparse.ArgumentParser(description="Unsloth Evaluation (Faster and can actually fit in 1 GPU).")
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
...(Step-by-step analysis of the log entry.)
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
    decoded = tokenizer.batch_decode(outputs)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    reference = entry["reference"]
    generated_responses.append(response)
    references.append(reference)
    with open("generated_outputs.txt", "a", encoding="utf-8") as f:
        f.write(f"-----------\n")
        f.write("Reference:\n")
        f.write(reference.strip() + "\n\n")
        f.write("Generated Response:\n")
        f.write(response.strip() + "\n")
        f.write("\n" + "=" * 50 + "\n\n")

st_model = SentenceTransformer("all-MiniLM-L6-v2")
for ref, pred in tqdm.tqdm(
    zip(references, generated_responses),
    total=len(references),
    desc="Evaluating Explanations",
):
    pattern = (
        r"^<thinking>(?P<thinking>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>$"
    )
    compiled = re.compile(pattern, re.DOTALL)
    match = compiled.search(pred)
    if not match:
        continue
    pattern_dict = match.groupdict()
    pred = pattern_dict["answer"]
    ref_label = multi_label(ref)
    pred_label = multi_label(pred)
    refs.append(ref_label)
    resps.append(pred_label)
    if ref_label == [0] or pred_label == [0]:
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
print(
    "Classification Report:\n",
    classification_report(y_true, y_pred, target_names=class_labels),
)
hamming = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hamming}")

average_score = np.mean(similarity_scores)
print(f"Similarity Score for Explanation:{average_score}")
