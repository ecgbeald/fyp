import pandas as pd

def load_csv(dataset_path):
    df = pd.read_csv(
        dataset_path,
        skiprows=lambda x: x in range(1),
        names=["log", "label", "category", "misc", "accept"],
    )
    return df
