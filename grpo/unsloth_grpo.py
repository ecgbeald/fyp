from unsloth import FastLanguageModel
import re
import os
from trl import GRPOConfig, GRPOTrainer
import datetime


def parse_response(text):
    pattern = r"""
    ^\{
    \s*"classification"\s*:\s*"(?P<classification>Malicious|Benign)"\s*,\s*
    "reason"\s*:\s*"\[?(?P<reason>[^\]]+)\]?"\s*,\s*
    "explanation"\s*:\s*"(?P<explanation>.*?)"
    \s*\}$
    """
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)
    if text.startswith("```json"):
        text = text.replace("```json", "").strip("` \n")
    elif text.startswith("```"):
        text = text.strip("` \n")

    match = regex.search(text)
    if not match:
        return None
    return match.groupdict()


def penalize_specific_words(text, penalty_per_word=1.0):
    words_to_penalize = [
        "check",
        "analyze",
        "examine",
        "access",
        "interpret",
        "understand",
    ]
    pattern = r"\b(" + "|".join(words_to_penalize) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    total_penalty = len(matches) * penalty_per_word
    return total_penalty


def parse_thinking_len(thinking, penalty_factor=2.0, max_url_len=100):
    # penalise long url
    total_penalty = 0.0
    url_pattern = r"https?://[^\s]+"
    url_matches = re.findall(url_pattern, thinking)
    for url in url_matches:
        url_len = len(url)
        if url_len > max_url_len:
            total_penalty += penalty_factor * (url_len - max_url_len)
    return total_penalty


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = 3):

    def generate_ngrams(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(text: str) -> float:
        if not text:
            return 0.0

        word_count = len(text.split())
        if word_count < ngram_size:
            return 0.0

        ngrams = set(generate_ngrams(text, ngram_size))
        total_ngrams = word_count - ngram_size + 1

        penalty = (1 - len(ngrams) / total_ngrams) * max_penalty

        return penalty

    return repetition_penalty_reward


def penalize_inferred_ip_reasoning(text, penalty=5.0):
    pattern = r"^(?=.*\bIP\b)(?=.*\b(?:\d{1,3}\.){3}\d{1,3}\b).*"
    lines = text.splitlines()
    for line in lines:
        if re.search(pattern, line, re.IGNORECASE):
            return penalty
    return 0.0


def match_keywords(text):
    keyword_variants = {
        "user agent": [r"user-agent", r"user agent"],
        "url": [r"url"],
        "referer": [r"refer", r"referrer"],
        "status code": [r"status code"],
        "request method": [
            r"GET",
            r"POST",
            r"CONNECT",
            r"HEAD",
            r"DELETE",
            r"OPTIONS",
            r"request method",
        ],
    }
    found = set()
    for norm_keyword, variants in keyword_variants.items():
        for variant in variants:
            if re.search(rf"\b{variant}\b", text, re.IGNORECASE):
                found.add(norm_keyword)
                break

    return float(max(5.0, len(found)))


def get_steps_reward(thinking, max_match):
    pattern = r"^[-*] "
    reward = 5.0
    match = len(re.findall(pattern, thinking, re.MULTILINE))
    if match > max_match:
        reward -= (match - max_match) * 2.0
    return min(reward, match * 5 / 3)


def format_reward_func(completions, reference, **kwargs):
    rewards = []
    for pred, ref in zip(completions, reference):
        pred = pred[0]["content"]
        # with open("log_resp.txt", "a", encoding="utf-8") as f:
        #     f.write("\n" + "=" * 80 + "\n")
        #     f.write("[Output]:" + pred)
        pattern = r"^<thinking>(?P<thinking>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>$"
        compiled = re.compile(pattern, re.DOTALL)
        match = compiled.search(pred)
        if not match:
            rewards.append(0.0)
            continue
        pattern_dict = match.groupdict()
        thinking = pattern_dict["thinking"]
        if thinking.startswith("\n"):
            thinking = thinking[1:]
        if thinking.endswith("\n"):
            thinking = thinking[:-1]
        reward = get_steps_reward(thinking, 7)
        reward += 2
        if reward == 2:
            # f.write("Parsing bullet point for completion failed.\n")
            rewards.append(reward)
            continue
        reward -= parse_thinking_len(thinking)
        reward -= penalize_inferred_ip_reasoning(thinking)
        reward -= penalize_specific_words(thinking)
        penalty_func = get_repetition_penalty_reward()
        reward -= penalty_func(thinking)
        answer = pattern_dict["answer"]
        if answer.startswith("\n"):
            answer = answer[1:]
        if answer.endswith("\n"):
            answer = answer[:-1]
            # f.write("[Reference]:\n" + ref.strip() + "\n")
        reward += 2
        reward += min(len(thinking) // 10, 5)
        reward += match_keywords(thinking)
        parse_completion = parse_response(answer)
        if parse_completion is None:
            # f.write("Parsing failed for answer.\n")
            rewards.append(reward)
            continue
        reward += 3
        parse_reference = parse_response(ref)
        if parse_completion["classification"] == parse_reference["classification"]:
            reward += 2
        if parse_completion["reason"] == parse_reference["reason"]:
            reward += 5
        explanation = parse_completion["explanation"]
        reward -= parse_thinking_len(explanation)
        reward -= penalty_func(explanation)
        rewards.append(reward)
    return rewards


def training_args(max_seq_length, save_path):
    return GRPOConfig(
        use_vllm=True,
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=max_seq_length,
        max_completion_length=512,
        num_train_epochs=1.0,
        save_steps=230,
        max_grad_norm=0.1,
        # dataloader_num_workers=4,
        # dataloader_pin_memory=True,
        report_to="none",  # Can use Weights & Biases
        output_dir=save_path,
    )


def train(model_path, dataset, max_seq_length=2048, lora_rank=64):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # use vllm
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    now = datetime.datetime.now()
    timestamp = now.strftime("%d%m_%H-%M")
    save_path = f"Qwen2.5-7B-GRPO_{timestamp}"

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        processing_class=tokenizer,
        reward_funcs=format_reward_func,
        args=training_args(max_seq_length, save_path),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
    return model, tokenizer
