import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import torch
from torch import nn
from shallow_net.shallow_net import ShallowNet
from utils.parse_log import parse_logs_into_df
from utils.convert_dataset import treat_dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    # For F1 score
    tp = 0
    fp = 0
    fn = 0
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in dataloader:
            preds = model(xb)
            loss = F.binary_cross_entropy(preds, yb.float())
            total_loss += loss.item() * yb.size(0)

            predicted = (preds >= 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

            tp += ((predicted == 1) & (yb == 1)).sum().item()
            fp += ((predicted == 1) & (yb == 0)).sum().item()
            fn += ((predicted == 0) & (yb == 1)).sum().item()
    # Avoid zero division
    accuracy = correct / total if total else 0
    avg_loss = total_loss / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(dataset_path, save_path="combined.pth"):
    dataset_path = glob.glob(f"{dataset_path}/*.csv")
    df = treat_dataset(dataset_path)
    df = parse_logs_into_df(df)
    df = df.drop(columns=["accept"])
    X_train, X_temp, y_train, y_temp = train_test_split(
        df[["request", "referer", "user_agent"]],
        df["label"],
        test_size=0.2,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    tfidf_request = TfidfVectorizer(max_features=1000)
    tfidf_referer = TfidfVectorizer(max_features=500)
    tfidf_ua = TfidfVectorizer(max_features=1000)

    X_train_request = tfidf_request.fit_transform(X_train["request"].fillna(""))
    X_train_referer = tfidf_referer.fit_transform(X_train["referer"].fillna(""))
    X_train_ua = tfidf_ua.fit_transform(X_train["user_agent"].fillna(""))
    X_test_request = tfidf_request.transform(X_test["request"].fillna(""))
    X_test_referer = tfidf_referer.transform(X_test["referer"].fillna(""))
    X_test_ua = tfidf_ua.transform(X_test["user_agent"].fillna(""))

    X_train_combined = hstack([X_train_request, X_train_referer, X_train_ua]).toarray()
    X_test_combined = hstack([X_test_request, X_test_referer, X_test_ua]).toarray()

    X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    input_dim = X_train_combined.shape[1]
    model = ShallowNet(input_dim)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_acc = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        metrics = evaluate(model, test_dl)

        if metrics["precision"] > best_acc:
            best_acc = metrics["precision"]
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={metrics['loss']:.4f}, "
            f"Val Acc={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    print(f"Best accuracy - test set: {best_acc:.4f}")

    torch.save(
        {
            "model": model,
            "vectorizer_request": tfidf_request,
            "vectorizer_referer": tfidf_referer,
            "vectorizer_ua": tfidf_ua,
        },
        save_path,
    )

    # Evaluation on validation set
    X_valid_request = tfidf_request.transform(X_val["request"].fillna(""))
    X_valid_referer = tfidf_referer.transform(X_val["referer"].fillna(""))
    X_valid_ua = tfidf_ua.transform(X_val["user_agent"].fillna(""))
    X_valid_combined = hstack([X_valid_request, X_valid_referer, X_valid_ua]).toarray()

    X_valid_tensor = torch.tensor(X_valid_combined, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    valid_ds = TensorDataset(X_valid_tensor, y_valid_tensor)
    valid_dl = DataLoader(valid_ds, batch_size=64)

    with torch.no_grad():
        print(evaluate(model, valid_dl))


def inference(
    request_text, referer_text, ua_text, model, tfidf_request, tfidf_referer, tfidf_ua, threshold=0.1
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
        label = int(prob >= threshold)

    return {"probability": prob, "label": label}