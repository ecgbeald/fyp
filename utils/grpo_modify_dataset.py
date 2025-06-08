from datasets import load_from_disk

grpo_format = """Please respond in the following format in English:
<thinking>
...(Provide an analysis of the following log entry, justify your reasoning using bullet point)
...(Do NOT analyse the IP address)
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

def modify_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(
        map_to_prompt,
        batched=True,
    )
    dataset["train"] = dataset["train"].remove_columns("messages")
    dataset["test"] = dataset["test"].remove_columns("messages")
    dataset["valid"] = dataset["valid"].remove_columns("messages")
    return dataset