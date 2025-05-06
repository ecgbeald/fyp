from datasets import load_dataset, DatasetDict
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="../data/dataset.hf",
    help="Path to the output dataset.",
)
parser.add_argument(
    "--json_path",
    type=str,
    default="../data/prompt.json",
    help="Path to the input JSON file.",
)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.json_path, split="train")
dataset = dataset.train_test_split(test_size=0.2)
test_valid = dataset["test"].train_test_split(test_size=0.5)
dataset = DatasetDict(
    {
        "train": dataset["train"],
        "test": test_valid["train"],
        "valid": test_valid["test"],
    }
)

dataset.save_to_disk(args.dataset_path)
