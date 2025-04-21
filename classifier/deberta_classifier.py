from transformers import DebertaV2Tokenizer, DataCollatorWithPadding, TrainingArguments, DebertaV2ForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import evaluate
import datetime
import glob
import pandas as pd

def load_csv(dataset_path):
    df = pd.read_csv(dataset_path, skiprows=lambda x: x in range(1), names=['log', 'label', 'category', 'misc', 'accept'])
    return df

def preprocess_function(examples):    
    examples['log'] = [str(text) for text in examples['log']]
    return tokenizer(examples['log'], truncation=True, padding=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']
    acc = acc_metric.compute(predictions=predictions, references=labels)['accuracy']
    return {"f1": f1, "accuracy": acc}

model_name = "microsoft/deberta-v3-large"
now = datetime.datetime.now()
timestamp = now.strftime("%d%m_%H-%M")

dataset_path = glob.glob("../data/*.csv")
df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
df = df.drop(df.columns[2:], axis=1)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
test_valid = dataset['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({'train': dataset['train'], 'test': test_valid['train'], 'valid': test_valid['test']})
dataset.save_to_disk("test.hf")

model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model.cuda()

tokenized_train = dataset['train'].map(preprocess_function, batched=True)
tokenized_test = dataset['test'].map(preprocess_function, batched=True)

args = TrainingArguments(
    output_dir=f"checkpoints/{timestamp}",
    learning_rate=2e-05,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=8,
    bf16=True,
    weight_decay=0.003,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    max_grad_norm=1.0,
    push_to_hub=False,
    report_to="none",
    # seed=42,
    # data_seed=42,
)

trainer = Trainer(
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

tokenized_valid = dataset['valid'].map(preprocess_function, batched=True)
trainer.train()
eval_res = trainer.evaluate(tokenized_valid)
print(eval_res)
