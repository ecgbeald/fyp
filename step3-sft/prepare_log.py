import pandas as pd
import glob
import json

def load_csv(dataset_path):
    df = pd.read_csv(dataset_path, skiprows=lambda x: x in range(1), names=['log', 'label', 'category', 'misc', 'accept'])
    return df

def generate_prompt(log):
    user_prompt = ("Given a log entry collected from an Apache HTTP server, classify it as either \"Malicious\" or \"Benign\".\n\n"
            "If the log is classified as malicious, specify the reason(s) (can be multiple) by selecting from the following categories: \n\n"
            "1. information exposure (reconaissance, scanning)\n"
            "2. injection (including command injection, sql injection, XML external entity attack, shellcode injection)\n"
            "3. path traversal\n"
            "4. remote code execution\n"
            "5. proxy-based attack (Server-Side Request Forgery, open redirect)\n"
            "6. cross site scripting\n"
            "7. local file inclusion\n"
            "8. prompt injection targeting LLM models\n"
            "9. other (not mentioned above, e.g., crypto mining, remote file inclusion, click spamming, etc.)\n\n"
            "Return your answer in strict JSON format for structured parsing. Use the following format:\n\n"
            "{{\n\"classification\": \"Malicious or Benign\",\n  \"reason\": \"Comma-separated list of category numbers if malicious; leave empty if benign\",\n"
            "\"Explaination\": why the weblog provided is malicious, leave empty if benign.\n}}\n"
        )    
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert analyzing Apache log entries to detect potential security threats."},
        {"role": "user", "content": user_prompt + "\nLog:" + log},
    ]
    return messages

def generate_response(label, category, explanation):
    if label == 0:
        return {"role": "assistant", "content": "```json {{\n\"classification\":\"Benign\",\n\"reason\":\"[0]\",\n\"explaination\":\"\"\n}}\n```"}
    else:
        return {"role": "assistant", "content": f"```json {{\n\"classification\":\"Malicious\",\n\"reason\":\"{str(category)}\",\n\"explaination\":\"{explanation}\"\n}}\n```"}

dataset_path = glob.glob("../data/fyp_data/*.csv")
taxonomy_map = {
    "info": 1,
    "injection": 2,
    "traversal": 3,
    "rce": 4,
    "proxy": 5,
    "xss": 6,
    "lfi": 7,
    "llm": 8,
    "other": 9
}

df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
df = df.drop(df.columns[3], axis=1)
df["category"] = df["category"].apply(    
    lambda x: sorted([
        taxonomy_map[k.strip().lower()]
        for k in str(x).split(',')
        if k.strip().lower() in taxonomy_map
    ]) if pd.notna(x) else [0]
    )

dicts = []
for _, row in df.iterrows():
    conversation = generate_prompt(row.iloc[0])
    conversation.append(generate_response(row.iloc[1], row.iloc[2],row.iloc[3]))
    entry = {"messages": conversation}
    dicts.append(entry)

print(f"Number of log entries: {len(df)}")
with open("../data/prompt.json", "w", encoding="utf-8") as f:
    json.dump(dicts, f, indent=2, ensure_ascii=False)