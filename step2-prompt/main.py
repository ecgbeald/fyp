from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
import numpy as np
import pandas as pd

# load dataset
dataset_path = "/vol/bitbucket/rm521/step2-prompt/log.csv"
df = pd.read_csv(dataset_path, names=['log', 'label'])

# load LLM
llm = LLM(model="/vol/bitbucket/rm521/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

labels = ["malicious", "not malicious"]


def generate_zero_shot(log):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Classify the following log as either 'malicious' or 'benign', make sure the classification is mentioned in your response:\n Log:" + log},
    ]
    return messages


def classify_log(log):
    message = generate_zero_shot(log)
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