import glob
from utils.convert_dataset import treat_dataset
from utils.prompt import generate_multiple_zero_shot, generate_mult_response
import json

def prepare_log(dataset_path, output_file, batch_size=1):
    dataset_path = glob.glob(f"{dataset_path}/*.csv")
    df = treat_dataset(dataset_path)
    dicts = []
    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i+batch_size]
        if len(rows) < batch_size:
            continue
        logs = [row[0] for _, row in rows.iterrows()]
        conversation = generate_multiple_zero_shot(logs)
        response_rows = [(row[1], row[2], row[3]) for _, row in rows.iterrows()]
        response = generate_mult_response(response_rows)
        conversation.append(response)
        entry = {"messages": conversation}
        dicts.append(entry)
        
    print(f"Number of log entries: {len(df)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2, ensure_ascii=False)
