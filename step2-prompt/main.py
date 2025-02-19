from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
import numpy as np
import pandas as pd

# load dataset
dataset_path = "/vol/bitbucket/rm521/step2-prompt/log.csv"
df = pd.read_csv(dataset_path, names=['log', 'label'])

# load LLM
llm = LLM(model="/vol/bitbucket/rm521/models/Qwen2.5-0.5B-Instruct")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/rm521/models/Qwen2.5-0.5B-Instruct")

labels = ["malicious", "not malicious"]


def generate_zero_shot(log):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Classify the following log as either 'malicious' or 'benign', make sure the classification is mentioned in your response:\n Log:" + log},
    ]
    return messages

def generate_few_shot(log):
    message = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Classify the following log as either 'malicious' or 'benign', make sure the classification is mentioned in your response:\n \
         Log: 192.168.1.110 - - [18/Feb/2025:10:25:20 +0000] ""POST /cgi-bin/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/bin/bash HTTP/1.1"" 200 210 ""-"" ""echo Content-Type: text/plain; echo; netstat -tulpn"", Classification: malicious,\n \
         Log: 192.168.1.103 - - [18/Feb/2025:10:18:45 +0000] ""POST /cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/bash HTTP/1.1"" 200 612 ""-"" ""echo Content-Type: text/plain; echo; whoami"", Classification: malicious, \n\
         Log: 192.168.1.204 - - [18/Feb/2025:12:06:40 +0000] ""GET /cart HTTP/1.1"" 200 768 ""-"" ""User-Agent: Chrome/99.0"", Classification: benign \
         Log:" + log + "Classification:"},
    ]
    return message


def classify_log(log):
    message = generate_few_shot(log)
    # Perform zero-shot classification
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([text], sampling_params)
    for output in outputs:
        generated_answer = output.outputs[0].text
        print(generated_answer)
        if "benign" in generated_answer.lower():
            return 0
    return 1
        # Parse the result (assuming it directly returns 'malicious' or 'not malicious')
        
    
df["prediction"] = df["log"].apply(classify_log)

correct_count = 0
rows = 0
for index, row in df.iterrows():
    rows += 1
    correct_count += row["label"] == row["prediction"]
    print(f"Prediction: {row['prediction']}")
    print(f"Label: {row['label']}")
    print("-----")
print(correct_count / rows)