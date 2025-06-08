# file for splitting dataset to 8-1-1 train, valid, test
from datasets import load_dataset, DatasetDict

def split_dataset_and_save(dataset_path, save_path):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    test_valid = dataset["test"].train_test_split(test_size=0.5)
    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "test": test_valid["train"],
            "valid": test_valid["test"],
        }
    )

    dataset.save_to_disk(save_path)
