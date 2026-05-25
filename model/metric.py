import numpy as np
import torch


def multiclass_metrics_from_confusion_matrix(cm: np.ndarray) -> dict:
    """
    Compute accuracy, macro/weighted precision, recall, F1 from confusion matrix.
    Rows = true class, columns = predicted class.
    """
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "per_class": {},
        }

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)

    precision = np.divide(tp, pred_counts, out=np.zeros(n_classes), where=pred_counts > 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes),
        where=(precision + recall) > 0,
    )

    weights = support / support.sum()
    per_class = {
        int(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    return {
        "accuracy": float(tp.sum() / total),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "per_class": per_class,
    }


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    k = min(k, output.size(1))  # max = number of classes
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def reconstruction_error(output, target):
    """Mean absolute reconstruction error for autoencoders."""
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()
