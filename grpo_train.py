from grpo.unsloth_grpo import train
from grpo.unsloth_eval_chat import eval
from utils.multi_label_bin import process_mult
from utils.grpo_modify_dataset import modify_dataset

if __name__ == "__main__":
    dataset_path = "data/dataset.hf"
    model_path = "/rds/general/user/rm521/home/fyp/grpo/Qwen2.5-7B-GRPO_Merged_2005_04-49"
    max_seq_length = 2048
    lora_rank = 64

    dataset = modify_dataset(dataset_path)

    model, tokenizer = train(model_path, dataset, max_seq_length, lora_rank)
    # to load model as model_path
    subset = dataset['valid']
    # references, generated_answers = eval(dataset, model_path, max_seq_length)
    references, generated_answers = eval(subset, model=model, tokenizer=tokenizer, max_seq_length=max_seq_length)
    
    process_mult(references, generated_answers)