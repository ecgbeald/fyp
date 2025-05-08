from sentence_transformers import SentenceTransformer, util
from label_parsing import multi_label, parse_explain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss
import numpy as np


def process_mult(references, generated_responses):
    similarity_scores = []
    refs = []
    resps = []
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    for ref, pred in zip(references, generated_responses):
        ref_label = multi_label(ref["content"])
        pred_label = multi_label(pred)
        refs.append(ref_label)
        resps.append(pred_label)
        if ref_label == [0] or pred_label == [0]:
            continue
        ref_exp = parse_explain(ref["content"])
        pred_exp = parse_explain(pred)
        ref_emb = st_model.encode(ref_exp, convert_to_tensor=True)
        pred_emb = st_model.encode(pred_exp, convert_to_tensor=True)
        sim_score = util.cos_sim(ref_emb, pred_emb).item()
        similarity_scores.append(sim_score)
        print(ref_exp)
        print(pred_exp)
        print("\n")

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(refs)
    y_pred = mlb.transform(resps)
    class_labels = [str(label) for label in mlb.classes_]
    print(class_labels)
    print(
        "Classification Report:\n",
        classification_report(y_true, y_pred, target_names=class_labels),
    )
    hamming = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hamming}")

    average_score = np.mean(similarity_scores)
    print(f"Similarity Score for Explanation:{average_score}")
