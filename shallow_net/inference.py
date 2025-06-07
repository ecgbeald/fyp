import torch
from scipy.sparse import hstack
import argparse
from utils import parse_log_line
from shallow_net import ShallowNet


def predict_single(
    request_text, referer_text, ua_text, model, tfidf_request, tfidf_referer, tfidf_ua
):
    X_req = tfidf_request.transform([request_text])
    X_ref = tfidf_referer.transform([referer_text])
    X_ua = tfidf_ua.transform([ua_text])

    X_combined = hstack([X_req, X_ref, X_ua]).toarray()

    X_tensor = torch.tensor(X_combined, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        prob = output.item()
        label = int(prob >= 0.1)

    return {"probability": prob, "label": label}


parser = argparse.ArgumentParser(description="Parse an Apache Log entry.")
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

combined = torch.load("shallow_net/combined.pth", weights_only=False)
model = combined["model"]
tfidf_request = combined["vectorizer_request"]
tfidf_referer = combined["vectorizer_referer"]
tfidf_ua = combined["vectorizer_ua"]

result = predict_single(
    example_request,
    example_referer,
    example_user_agent,
    model,
    tfidf_request,
    tfidf_referer,
    tfidf_ua,
)
print(f"Probability: {result['probability']:.4f}, Label: {result['label']}")
