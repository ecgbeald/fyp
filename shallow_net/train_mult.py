import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack
import torch
from shallow_net.shallow_net import ShallowMultiLabelNet
from utils.convert_dataset import treat_dataset
from utils.parse_log import parse_logs_into_df
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = F.binary_cross_entropy(preds, yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            preds = model(xb)
            loss = F.binary_cross_entropy(preds, yb.float())
            total_loss += loss.item() * xb.size(0)
            all_preds.append((preds >= threshold).int())
            all_targets.append(yb.int())

    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    samples_f1 = f1_score(y_true, y_pred, average="samples")

    class_report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "val_loss": total_loss / len(dataloader.dataset),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples_f1": samples_f1,
        "classification_report": class_report,
    }


def train(dataset_path, save_path="mult.pth"):
    dataset_path = glob.glob(f"{dataset_path}/*.csv")
    df = treat_dataset(dataset_path)
    df = parse_logs_into_df(df)
    df = df.drop(columns=["accept"])
    mlb = MultiLabelBinarizer()
    multi_hot = mlb.fit_transform(df["category"])
    X_train, X_temp, y_train, y_temp = train_test_split(
        df[["request", "referer", "user_agent"]],
        multi_hot,
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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_dim = X_train_combined.shape[1]
    output_dim = y_train.shape[1]
    model = ShallowMultiLabelNet(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    epochs = 10
    best_acc = 42
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dl, optimizer)
        metrics = evaluate(model, test_dl)
        
        if metrics["val_loss"] < best_acc:
            best_acc = metrics["val_loss"]
            torch.save(model.state_dict(), "best_model.pth")


        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={metrics['val_loss']:.4f}, "
            f"Micro F1={metrics['micro_f1']:.4f}, "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Weighted F1={metrics['weighted_f1']:.4f}, "
            f"Samples F1={metrics['samples_f1']:.4f}"
        )
        # print(metrics['classification_report'])

    torch.save(
        {
            "model": model,
            "vectorizer_request": tfidf_request,
            "vectorizer_referer": tfidf_referer,
            "vectorizer_ua": tfidf_ua,
        },
        save_path,
    )

    X_valid_request = tfidf_request.transform(X_val["request"].fillna(""))
    X_valid_referer = tfidf_referer.transform(X_val["referer"].fillna(""))
    X_valid_ua = tfidf_ua.transform(X_val["user_agent"].fillna(""))
    X_valid_combined = hstack([X_valid_request, X_valid_referer, X_valid_ua]).toarray()

    X_valid_tensor = torch.tensor(X_valid_combined, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_val, dtype=torch.float32)

    valid_ds = TensorDataset(X_valid_tensor, y_valid_tensor)
    valid_dl = DataLoader(valid_ds, batch_size=64)

    with torch.no_grad():
        print(evaluate(model, valid_dl))

# return the probability and predicted labels for the input log
def inference(
    request_text, referer_text, ua_text, model, tfidf_request, tfidf_referer, tfidf_ua, threshold=0.6
):
    X_req = tfidf_request.transform([request_text])
    X_ref = tfidf_referer.transform([referer_text])
    X_ua = tfidf_ua.transform([ua_text])

    X_combined = hstack([X_req, X_ref, X_ua]).toarray()
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        probs = output.squeeze(0).numpy()
        preds = (probs >= threshold).astype(int)
    
        # if all(p < 0.7 for p in probs):
        #     return {"probability": 0.0, "label": "unknown"}


    return probs, preds
