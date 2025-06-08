import pandas as pd
import glob
import json
import argparse

parser = argparse.ArgumentParser(description="Process Apache log entries.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="../data/fyp_data/*.csv",
    help="Path to the dataset (supports glob patterns).",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="../data/mult.json",
    help="Path to the output JSON file.",
)
parser.add_argument(
    "--split", type=bool, default=False, help="Flag to split the message roles."
)
args = parser.parse_args()


def load_csv(dataset_path):
    df = pd.read_csv(
        dataset_path,
        skiprows=lambda x: x in range(1),
        names=["log", "label", "category", "misc", "accept"],
    )
    return df


def generate_prompt(log):
    user_prompt = (
        'Given a log entry collected from an Apache HTTP server, classify it as either "Malicious" or "Benign".\n\n'
        "If the log is classified as malicious, specify the reason(s) (can be multiple) by selecting from the following categories: \n\n"
        "1. information exposure (reconnaissance, scanning)\n"
        "2. injection (including command injection, sql injection, XML external entity attack, shellcode injection)\n"
        "3. path traversal\n"
        "4. remote code execution\n"
        "5. proxy-based attack (Server-Side Request Forgery, open redirect)\n"
        "6. cross site scripting\n"
        "7. local file inclusion\n"
        "8. prompt injection targeting LLM models\n"
        "9. other (not mentioned above, e.g., crypto mining, remote file inclusion, click spamming, etc.)\n\n"
        "Return your answer in strict JSON format for structured parsing. Use the following format:\n\n"
        '{\n"classification": "Malicious or Benign",\n"reason": "Comma-separated list of category numbers if malicious; return [0] if benign",\n'
        '"Explanation": why the weblog provided is malicious, leave this field empty if the log is benign.\n}\n'
    )
    if not args.split:
        messages = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.",
            },
            {"role": "user", "content": user_prompt + "\nLog:" + log},
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.",
            },
            {"role": "instruction", "content": user_prompt},
            {"role": "input", "content": log},
        ]
    return messages


def generate_mult_prompt(logs):
    user_prompt = (
        'Given a log entry collected from an Apache HTTP server, classify it as either "Malicious" or "Benign".\n\n'
        "If the log is classified as malicious, specify the reason(s) (can be multiple) by selecting from the following categories: \n\n"
        "1. information exposure (reconnaissance, scanning)\n"
        "2. injection (including command injection, sql injection, XML external entity attack, shellcode injection)\n"
        "3. path traversal\n"
        "4. remote code execution\n"
        "5. proxy-based attack (Server-Side Request Forgery, open redirect)\n"
        "6. cross site scripting\n"
        "7. local file inclusion\n"
        "8. prompt injection targeting LLM models\n"
        "9. other (not mentioned above, e.g., crypto mining, remote file inclusion, click spamming, etc.)\n\n"
        "Return your answer in strict JSON format for structured parsing. Use the following format:\n\n"
        '{\n"classification": "Malicious or Benign",\n"reason": "Comma-separated list of category numbers if malicious; return [0] if benign",\n'
        '"Explanation": why the weblog provided is malicious, leave this field empty if the log is benign.\n}\n'
    )
    if not args.split:
        messages = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.",
            },
            {"role": "user", "content": user_prompt + "\nLogs:" + '\n'.join(str(log) for log in logs)},
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert analyzing Apache log entries to detect potential security threats.",
            },
            {"role": "instruction", "content": user_prompt},
            {"role": "input", "content": '\n'.join(str(log) for log in logs)},
        ]
    return messages



def generate_response(label, category, explanation):
    if label == 0:
        return {
            "role": "assistant",
            "content": f'```json {{\n"classification":"Benign",\n"reason":"[0]",\n"explanation":""\n}}\n```',
        }
    else:
        return {
            "role": "assistant",
            "content": f'```json {{\n"classification":"Malicious",\n"reason":"{str(category)}",\n"explanation":"{explanation}"\n}}\n```',
        }

def generate_mult_response(rows):
    resp = []
    for label, cat, expl in rows:
        if label == 0:
            resp.append(f'{{\n"classification":"Benign",\n"reason":"[0]",\n"explanation":""\n}}\n')
        else:
            resp.append(f'{{\n"classification":"Malicious",\n"reason":"{str(cat)}",\n"explanation":"{expl}"\n}}\n')
    return {
        "role": "assistant",
        "content" : ''.join(str(res) for res in resp),
    }

dataset_path = glob.glob(args.dataset_path)
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

df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
df = df.drop(df.columns[3], axis=1)
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

dicts = []
for i in range(0, len(df), 3):
    rows = df.iloc[i:i+3]
    if len(rows) < 3:
        continue  # Skip if there aren't 3 rows left

    # Get the first value from the first row
    logs = [row[0] for _, row in rows.iterrows()]
    conversation = generate_mult_prompt(logs)
    
    response_rows = [(row[1], row[2], row[3]) for _, row in rows.iterrows()]
    response = generate_mult_response(response_rows)
    conversation.append(response)

    entry = {"messages": conversation}
    dicts.append(entry)
# for _, row in df.iterrows():
#     conversation = generate_prompt(row.iloc[0])
#     conversation.append(generate_response(row.iloc[1], row.iloc[2], row.iloc[3]))
#     entry = {"messages": conversation}
#     dicts.append(entry)

print(f"Number of log entries: {len(df)}")
with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(dicts, f, indent=2, ensure_ascii=False)
