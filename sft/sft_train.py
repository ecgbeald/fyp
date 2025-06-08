# for submitting script job
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import datetime

def create_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM",
    )
    
def training_args(save_path, max_seq_length):
    return SFTConfig(
        output_dir=save_path,
        load_best_model_at_end=True,
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
        num_train_epochs=3.0,
        bf16=True,
        max_seq_length=max_seq_length,
        save_steps=2000,
        eval_strategy="steps",
        eval_steps=500,
    )


def setup_trainer(model, dataset, peft_config, sft_config):
    return SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
        peft_config=peft_config,
    )
    

def train(model_path, dataset, max_seq_len):
    max_memory = {0: torch.cuda.get_device_properties(0).total_memory}
    now = datetime.datetime.now()
    timestamp = now.strftime("%d%m_%H-%M")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory
    )
    model_name = model.config._name_or_path.split("/")[-1]
    save_path = f'{model_name}-SFT_{timestamp}'
    peft_config = create_peft_config()
    sft_config = training_args(save_path, max_seq_len)
    trainer = setup_trainer(model, dataset, peft_config, sft_config)
    trainer.train()
    trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(f"{save_path}_Merged")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(f"{save_path}_Merged")
    return save_path