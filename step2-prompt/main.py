from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

def load_csv(dataset_path):
    df = pd.read_csv(dataset_path, skiprows=lambda x: x in range(1), names=['log', 'label', 'category'])
    return df

def drop_category(df):
    df = df.drop(columns=['category'])
    return df

def generate_zero_shot(log):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Classify the following log as either 'malicious' or 'benign', make sure the classification is mentioned in your response:\n Log:" + log},
    ]
    return messages

def generate_few_shot(log):
    message = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Classify the last log entry as either 'malicious' or 'benign', the previous log entries and classification is for you to study:\n \
         Log: 192.168.1.110 - - [18/Feb/2025:10:25:20 +0000] ""POST /cgi-bin/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/.%2e/bin/bash HTTP/1.1"" 200 210 ""-"" ""echo Content-Type: text/plain; echo; netstat -tulpn"", Classification: malicious,\n \
         Log: 192.168.1.103 - - [18/Feb/2025:10:18:45 +0000] ""POST /cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/bash HTTP/1.1"" 200 612 ""-"" ""echo Content-Type: text/plain; echo; whoami"", Classification: malicious, \n\
         Log: 192.168.1.204 - - [18/Feb/2025:12:06:40 +0000] ""GET /cart HTTP/1.1"" 200 768 ""-"" ""User-Agent: Chrome/99.0"", Classification: benign \
         Log:" + log + "Classification:"},
    ]
    return message


def classify_log(prompts):
    # Perform zero-shot classification
    predictions = []
    text = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate(text, sampling_params)
    for output in outputs:
        generated_answer = output.outputs[0].text
        print(generated_answer)
        if "benign" in generated_answer.lower():
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

if __name__ == "__main__":
    dataset_path = "/vol/bitbucket/rm521/fyp/step2-prompt/logs.csv"
    df = load_csv(dataset_path)
    df = drop_category(df)
    
    dataset = Dataset.from_pandas(df)
    
    llm = LLM(model="/vol/bitbucket/rm521/models/Qwen2.5-7B-Instruct")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/rm521/models/Qwen2.5-7B-Instruct")
    labels = ["malicious", "benign"]
    
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    predictions = []
    for batch in dataloader:
        prompts = []
        for log in batch['log']:
            prompts.append(generate_few_shot(log))        
        predictions.extend(classify_log(prompts))
    
    df['prediction'] = predictions
    print(f"Len: {len(predictions)}, len of df: {len(df)}")
    print(df)
    
    correct_count = (df['label'] == df['prediction']).sum()
    total_count = len(df)
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy}")

    
    