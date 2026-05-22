"""
Plot a confusion matrix PDF in the same style as test.py (Blues heatmap, annotated cells).

Usage examples:
  python scripts/plot_confusion_matrix.py --demo -o confusion_matrix.pdf
  python scripts/plot_confusion_matrix.py --y-true y_true.npy --y-pred y_pred.npy -o out.pdf
  python scripts/plot_confusion_matrix.py --csv preds.csv -o out.pdf
  python scripts/plot_confusion_matrix.py --matrix "8,1,1;2,10,0;0,2,8" --labels Kočka,Pes,Myš -o out.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_loader.data_loaders as module_data

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "confusion_matrix.pdf"


# =====================================================
# =========           Functions               =========
# =====================================================

def build_confusion_matrix(y_true, y_pred):
    """Build NxN confusion matrix from flat prediction/target arrays."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred length mismatch: {y_true.size} vs {y_pred.size}")

    max_true = int(y_true.max()) if y_true.size else 0
    max_pred = int(y_pred.max()) if y_pred.size else 0
    n_classes = max(max_true, max_pred) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm, n_classes


def resolve_class_labels(n_classes, labels=None):
    """Human-readable class names (IoV dict or custom list)."""
    if labels is not None:
        if len(labels) != n_classes:
            raise ValueError(f"Expected {n_classes} labels, got {len(labels)}")
        return list(labels)

    class_labels = [str(i) for i in range(n_classes)]
    if n_classes == len(module_data.MULTICLASS_DICT):
        inv_map = {v: k for k, v in module_data.MULTICLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]
    elif n_classes == len(module_data.BINCLASS_DICT):
        inv_map = {v: k for k, v in module_data.BINCLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]
    return class_labels


def save_confusion_matrix(cm, class_labels, output_path, title="Matice záměn"):
    """Render confusion matrix as PDF (same style as test.py)."""
    n_classes = len(class_labels)
    cm = np.asarray(cm, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel="Skutečná třída",
        xlabel="Predikovaná třída",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Uloženo: {output_path.resolve()}")


def parse_matrix_string(matrix_str):
    """Parse '8,1,1;2,10,0;0,2,8' into 2D numpy array."""
    rows = []
    for row in matrix_str.strip().split(";"):
        row = row.strip()
        if not row:
            continue
        rows.append([int(x.strip()) for x in row.split(",")])
    if not rows:
        raise ValueError("Empty matrix string")
    cm = np.array(rows, dtype=np.int64)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Matrix must be square")
    return cm


def load_labels_from_csv(csv_path, true_col="true", pred_col="pred"):
    """Load integer label columns from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{true_col}' and '{pred_col}'")
    y_true = df[true_col].to_numpy(dtype=np.int64)
    y_pred = df[pred_col].to_numpy(dtype=np.int64)
    return y_true, y_pred


def demo_matrix():
    """Ukázková 3-třídní matice (Kočka, Pes, Myš)."""
    cm = np.array([[7, 1, 1], [4, 10, 2], [0, 1, 8]], dtype=np.int64)
    labels = ["Kočka", "Pes", "Myš"]
    return cm, labels


# =====================================================
# =========           Entry point             =========
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix PDF (same style as test.py)"
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--demo", action="store_true", help="ukázková 3-třídní matice (Kočka, Pes, Myš)")
    src.add_argument("--matrix", type=str, help="square matrix: rows separated by ';', values by ','")
    src.add_argument("--y-true", type=str, help="path to .npy/.txt with true class indices")
    src.add_argument("--y-pred", type=str, help="path to .npy/.txt with predicted class indices")
    src.add_argument("--csv", type=str, help="CSV with true/pred columns")

    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT), help="output PDF path")
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="comma-separated class names (e.g. BENIGN,DOS,GAS,...)",
    )
    parser.add_argument("--title", type=str, default="Matice záměn", help="nadpis grafu")
    parser.add_argument("--true-col", type=str, default="true", help="CSV column for true labels")
    parser.add_argument("--pred-col", type=str, default="pred", help="CSV column for predicted labels")
    args = parser.parse_args()
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]

    has_input = args.demo or args.matrix or args.csv or (args.y_true and args.y_pred)
    if not has_input:
        args.demo = True

    if args.demo:
        cm, labels = demo_matrix()
        save_confusion_matrix(cm, labels, args.output, title=args.title)
        return

    if args.matrix:
        cm = parse_matrix_string(args.matrix)
        class_labels = resolve_class_labels(cm.shape[0], labels)
        save_confusion_matrix(cm, class_labels, args.output, title=args.title)
        return

    if args.csv:
        y_true, y_pred = load_labels_from_csv(args.csv, args.true_col, args.pred_col)
        cm, n_classes = build_confusion_matrix(y_true, y_pred)
        class_labels = resolve_class_labels(n_classes, labels)
        save_confusion_matrix(cm, class_labels, args.output, title=args.title)
        return

    if args.y_true and args.y_pred:
        y_true = np.load(args.y_true) if args.y_true.endswith(".npy") else np.loadtxt(args.y_true, dtype=np.int64)
        y_pred = np.load(args.y_pred) if args.y_pred.endswith(".npy") else np.loadtxt(args.y_pred, dtype=np.int64)
        cm, n_classes = build_confusion_matrix(y_true, y_pred)
        class_labels = resolve_class_labels(n_classes, labels)
        save_confusion_matrix(cm, class_labels, args.output, title=args.title)
        return

    parser.error("Zadej --y-true a --y-pred společně, nebo jiný režim (--demo, --matrix, --csv).")


if __name__ == "__main__":
    main()
