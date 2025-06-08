import pandas as pd
from utils.load_csv import load_csv

def treat_dataset(dataset_path):
    taxonomy_map = {
        "info": 1,
        "injection": 2,
        "traversal": 3,
        "rce": 4,
        "proxy": 5,
        "xss": 6,
        "lfi": 7,
        "llm": 8,
        "other": 9,
    }
        
    df = pd.concat((load_csv(f) for f in dataset_path), ignore_index=True)
    df = df.drop(df.columns[3], axis=1)
    df["category"] = df["category"].apply(
        lambda x: (
            sorted(
                [
                    taxonomy_map[k.strip().lower()]
                    for k in str(x).split(",")
                    if k.strip().lower() in taxonomy_map
                ]
            )
            if pd.notna(x)
            else [0]
        )
    )
    return df
