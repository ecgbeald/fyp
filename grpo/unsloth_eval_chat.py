from unsloth import FastLanguageModel
import torch

import re
import tqdm

def load_model(model_path, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def eval(dataset, model_path=None, model=None, tokenizer=None, max_seq_length=2048):
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Either model_path or model/tokenizer must be provided.")
        model, tokenizer = load_model(model_path, max_seq_length)
    
    generated_responses = []
    references = []
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
        decoded = tokenizer.batch_decode(outputs)[0]
        match = re.search(r"^Log Entry:.*$", decoded, re.MULTILINE)
                
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        reference = {"content": entry["reference"]}
        generated_responses.append(response)
        references.append(reference)

    generated_answers = []
    for pred in generated_responses:
        pattern = (
            r"^<thinking>(?P<thinking>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>$"
        )
        compiled = re.compile(pattern, re.DOTALL)
        match = compiled.search(pred)
        if not match:
            generated_answers.append("")
            continue
        pattern_dict = match.groupdict()
        generated_answers.append(pattern_dict["answer"])

    assert len(generated_answers) == len(references)
    return references, generated_answers
