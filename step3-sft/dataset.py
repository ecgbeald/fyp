from datasets import load_dataset, DatasetDict

dataset = load_dataset("json", data_files="../data/prompt.json", split='train')
dataset = dataset.train_test_split(test_size=0.2)
test_valid = dataset['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({'train': dataset['train'], 'test': test_valid['train'], 'valid': test_valid['test']})

dataset.save_to_disk("../data/dataset.hf")