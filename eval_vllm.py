# Program for Evaluation on full dataset using VLLM

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import re
import glob
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from utils.label_parsing import parse_label_string
from utils.load_csv import load_csv
from utils.prompt import generate_few_shot
from utils.convert_dataset import treat_dataset


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


def eval_accuracy(df):
    total_count = len(df)
    correct_rows = df.apply(
        lambda row: (row["label"] == 1 and row["prediction"] != [0])
        or (row["label"] == 0 and row["prediction"] == [0]),
        axis=1,
    )
    correct_count = correct_rows.sum()
    return correct_count / total_count


def eval_multi_label(df):
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(df["category"])
    y_pred = mlb.transform(df["prediction"])
    class_labels = [str(label) for label in mlb.classes_]
    cr = classification_report(y_true, y_pred, target_names=class_labels)
    return cr


if __name__ == "__main__":
    # env variables to run VLLM on hpc environment
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="The path to the model directory", required=True
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The path to the dataset directory",
        default="data/fyp_data",
    )

    args = parser.parse_args()
    if not os.path.isdir(args.dataset_path):
        print(f"Error: {args.dataset_path} is not a valid directory.")
        exit(1)

    dataset_path = glob.glob(f"{args.dataset_path}/*.csv")
    df = treat_dataset(dataset_path)
    dataset = Dataset.from_pandas(df)

    llm = LLM(model=args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataloader = DataLoader(dataset["log"], batch_size=10, shuffle=False)
    predictions = []
    for batch in dataloader:
        prompts = []
        for log in batch:
            prompts.append(generate_few_shot(log))
        predictions.extend(classify_log(prompts))

    # Ensure predictions match the length of the dataset
    assert len(predictions) == len(df)
    df["prediction"] = predictions

    print(f"Accuracy of Binary Classification: {eval_accuracy(df)}\n")
    print(f"Classification Report:\n{eval_multi_label(df)}")
