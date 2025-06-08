# pipeline for sft training and evaluation, includes dataset preparation
# during development, these steps are run separately due to GPU limitations

from datasets import load_from_disk

from sft.eval import eval
from sft.sft_train import train

from utils.prepare_log import prepare_log
from utils.split_dataset import split_dataset_and_save
from utils.multi_label_bin import process_mult
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training and Evaluation Pipeline")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for log preparation")
    parser.add_argument("--model", type=str, help="Path to base model for SFT")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset directory",
        default="data/fyp_data",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="data/myset",
        help="Mode for training",
    )
    args = parser.parse_args()

    prepare_log(args.dataset_path, "tmp.json", args.batch_size)
    split_dataset_and_save("tmp.json", args.dataset_save_path)

    dataset = load_from_disk(f"{args.dataset_save_path}.hf")
    save_path = train(
        model_path=args.model,
        dataset=dataset,
        max_seq_len=8192,
    )

    subset = dataset['valid']
    generated_responses, references = eval(subset, model_path=save_path, batch_size=5)
    assert len(references) == len(generated_responses)
    process_mult(references, generated_responses)