from transformers import DebertaV2Tokenizer, DataCollatorWithPadding, TrainingArguments, DebertaV2ForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import evaluate
import datetime
import glob
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

def load_csv(dataset_path):
    df = pd.read_csv(dataset_path, skiprows=lambda x: x in range(1), names=['log', 'label', 'category', 'misc', 'accept'])
    return df

def preprocess_function_bin(examples):    
    examples['log'] = [str(text) for text in examples['log']]
    return tokenizer(examples['log'], truncation=True, padding=True, max_length=512)

def make_preprocess_function(label_cols):
    def preprocess_function_mult(examples):    
        examples['log'] = [str(text) for text in examples['log']]
        tokenized = tokenizer(
            examples['log'],
            truncation=True,
            padding="max_length",
            max_length=1024
        )
        labels = list(zip(*(examples[str(label)] for label in label_cols)))
        tokenized["labels"] = [list(map(float, l)) for l in labels]
        return tokenized
    return preprocess_function_mult

def compute_metrics_bin(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']
    acc = acc_metric.compute(predictions=predictions, references=labels)['accuracy']
    return {"f1": f1, "accuracy": acc}

def compute_metrics_mult(eval_pred):
    predictions, labels = eval_pred
    predictions = 1 / (1 + np.exp(-predictions))  # apply sigmoid
    predictions = (predictions > 0.5).astype(int)
    labels = labels.astype(int)
    f1_metric = evaluate.load("f1")
    
    return {
        "micro_f1": f1_score(labels, predictions, average="micro"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "samples_f1": f1_score(labels, predictions, average="samples"),
    }
    
def load_data(dataset_path, multi):
    dataset_path = glob.glob(dataset_path)
    df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
    if multi:
        df = df.drop(['label', 'misc', 'accept'], axis=1)
        taxonomy_map = {
            "info": 1,
            "injection": 2,
            "traversal": 3,
            "rce": 4,
            "proxy": 5,
            "xss": 6,
            "lfi": 7,
            "llm": 8,
            "other": 9
        }
        df["category"] = df["category"].apply(    
            lambda x: sorted([
                taxonomy_map[k.strip().lower()]
                for k in str(x).split(',')
                if k.strip().lower() in taxonomy_map
            ]) if pd.notna(x) else [0]
        )
        mlb = MultiLabelBinarizer()
        label_matrix = mlb.fit_transform(df['category'])
        label_columns = mlb.classes_
        for i, col in enumerate(label_columns):
            df[col] = label_matrix[:, i]
        label_cols = list(label_columns)
        df = df.drop(columns=['category'])
        return (df, label_columns)
    else:
        df = df.drop(df.columns[2:], axis=1)
        return (df, None)

def split_dataset(dataset):
    dataset = dataset.train_test_split(test_size=0.2)
    test_valid = dataset['test'].train_test_split(test_size=0.5)
    dataset = DatasetDict({'train': dataset['train'], 'test': test_valid['train'], 'valid': test_valid['test']})
    dataset.save_to_disk("test.hf")
    return dataset


if __name__ == "__main__":
    model_name = "microsoft/deberta-v3-large"
    now = datetime.datetime.now()
    timestamp = now.strftime("%d%m_%H-%M")
    dataset_path = "../data/fyp_data/*.csv"
    multi = False
    (df, label_columns) = load_data(dataset_path, multi)
    
    dataset = Dataset.from_pandas(df)
    dataset = split_dataset(dataset)

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    if multi:
        model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=len(label_columns), output_attentions=False, output_hidden_states=False, problem_type="multi_label_classification")
        preprocess_function = make_preprocess_function(list(label_columns))
        tokenized_train = dataset['train'].map(preprocess_function, batched=True)
        tokenized_test = dataset['test'].map(preprocess_function, batched=True)
        tokenized_valid = dataset['valid'].map(preprocess_function, batched=True)
    else:
        model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
        tokenized_train = dataset['train'].map(preprocess_function_bin, batched=True)
        tokenized_test = dataset['test'].map(preprocess_function_bin, batched=True)
        tokenized_valid = dataset['valid'].map(preprocess_function_bin, batched=True)
    model.cuda()
    
    args = TrainingArguments(
        output_dir=f"checkpoints/{timestamp}",
        learning_rate=2e-05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
    if multi:
        trainer = Trainer(
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            compute_metrics=compute_metrics_mult,
        )
    else:
        trainer = Trainer(
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            compute_metrics=compute_metrics_bin,
        )
    trainer.train()
    eval_res = trainer.evaluate(tokenized_valid)
    print(eval_res)
