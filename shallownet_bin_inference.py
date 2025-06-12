# shallow_net inference: for binary classification
import torch
import argparse
from utils.parse_log import parse_log_line
from shallow_net.shallow_net import ShallowNet
from shallow_net.train_bin import inference

parser = argparse.ArgumentParser(description="Parse an Apache Log entry.")
parser.add_argument(
    "--model", type=str, help="The shallownet combined model file"
)
parser.add_argument("log_line", type=str, help="The log line to parse")

args = parser.parse_args()

parsed_data = parse_log_line(args.log_line)
if "error" in parsed_data:
    print(
        f"Error parsing log line {parsed_data['log']}\nReason: {parsed_data['error']}\n"
    )
    exit(1)
example_request = parsed_data["request"]
example_referer = parsed_data["referer"]
example_user_agent = parsed_data["user_agent"]

combined = torch.load(f"{args.model}", weights_only=False)
model = combined["model"]
tfidf_request = combined["vectorizer_request"]
tfidf_referer = combined["vectorizer_referer"]
tfidf_ua = combined["vectorizer_ua"]

result = inference(
    example_request,
    example_referer,
    example_user_agent,
    model,
    tfidf_request,
    tfidf_referer,
    tfidf_ua,
    threshold=0.1,
)
print(f"Probability: {result['probability']:.4f}, Label: {result['label']}")
