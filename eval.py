from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import re
import glob
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss

from utils.label_parsing import parse_label_string
from utils.load_csv import load_csv
from utils.prompt import generate_few_shot

def classify_log(prompts):
    predictions = []
    text = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    outputs = llm.generate(text, sampling_params)
    for output in outputs:
        generated_answer = output.outputs[0].text.lower()
        matched = False
        for line in generated_answer.split("\n"):
            if '"reason":' in line:
                matched = True
                match = re.search(r'"reason":\s*(\[[^\]]*\]|"[^"]*"|\d+)', line)
                if match:
                    reason_str = match.group(1)
                    # Remove quotes if value is a string representation
                    if reason_str.startswith('"') and reason_str.endswith('"'):
                        reason_str = reason_str[1:-1]

                    categories = parse_label_string(reason_str)
                    predictions.append(categories)
                    break
                else:
                    predictions.append([0])
                    break
        if not matched:
            predictions.append([0])
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="The path to the model directory", required=True)
    parser.add_argument("dataset_path", type=str, help="The path to the dataset directory", required=True, default="data/fyp_data")
    
    args = parser.parse_args()
    taxonomy_map = {
        "info": 1,
        "injection": 2,
        "traversal": 3,
        "rce": 4,
        "proxy": 5,
        "xss": 6,
        "lfi": 7,
        "llm": 8,
        "other": 9,
    }
    dataset_path = glob.glob(f"{args.dataset_path}/*.csv")

    df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
    df = df.drop(df.columns[3:], axis=1)
    df["category"] = df["category"].apply(
        lambda x: (
            sorted(
                [
                    taxonomy_map[k.strip().lower()]
                    for k in str(x).split(",")
                    if k.strip().lower() in taxonomy_map
                ]
            )
            if pd.notna(x)
            else [0]
        )
    )
    dataset = Dataset.from_pandas(df)

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model=args.model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataloader = DataLoader(dataset["log"], batch_size=10, shuffle=False)
    predictions = []
    for batch in dataloader:
        prompts = []
        for log in batch:
            prompts.append(generate_few_shot(log))
        predictions.extend(classify_log(prompts))

    df["prediction"] = predictions

    total_count = len(df)
    correct_rows = df.apply(
        lambda row: (row["label"] == 1 and row["prediction"] != [0])
        or (row["label"] == 0 and row["prediction"] == [0]),
        axis=1,
    )
    correct_count = correct_rows.sum()
    accuracy = correct_count / total_count
    print(f"Accuracy of Binary Classification: {accuracy}")

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(df["category"])
    y_pred = mlb.transform(df["prediction"])
    class_labels = [str(label) for label in mlb.classes_]
    print(class_labels)
    print(
        "Classification Report:\n",
        classification_report(y_true, y_pred, target_names=class_labels),
    )
    hamming = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hamming}")
