import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from params import DEFAULT_CONFIG_PATH, DEFAULT_MODEL

# =====================================================
# =========       Constants and options       =========
# =====================================================

TEST_BATCH_SIZE = 512
TEST_NUM_WORKERS = 2


# =====================================================
# =========           Functions               =========
# =====================================================

def build_confusion_matrix(y_true, y_pred):
    """Build NxN confusion matrix from flat prediction/target arrays."""
    max_true = int(y_true.max()) if y_true.size else 0
    max_pred = int(y_pred.max()) if y_pred.size else 0
    n_classes = max(max_true, max_pred) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm, n_classes


def get_class_labels(n_classes):
    """Resolve human-readable class labels from data_loader dictionaries."""
    class_labels = [str(i) for i in range(n_classes)]

    if n_classes == len(module_data.MULTICLASS_DICT):
        inv_map = {v: k for k, v in module_data.MULTICLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]
    elif n_classes == len(module_data.CAR_MULTICLASS_DICT):
        inv_map = {v: k for k, v in module_data.CAR_MULTICLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]
    elif n_classes == len(module_data.BINCLASS_DICT):
        inv_map = {v: k for k, v in module_data.BINCLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]

    return class_labels


def save_confusion_matrix(cm, n_classes, class_labels, output_path):
    """Render confusion matrix as PDF and save."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def log_test_results(logger, metrics: dict, timing: dict, class_labels: list[str]) -> None:
    """Log classification metrics and inference timing (same style as autoencoder test)."""
    logger.info("Test results:")
    logger.info(f"    {'accuracy':25s}: {metrics['accuracy']:.6f}")
    logger.info(f"    {'precision_macro':25s}: {metrics['precision_macro']:.6f}")
    logger.info(f"    {'recall_macro':25s}: {metrics['recall_macro']:.6f}")
    logger.info(f"    {'f1_macro':25s}: {metrics['f1_macro']:.6f}")
    logger.info(f"    {'precision_weighted':25s}: {metrics['precision_weighted']:.6f}")
    logger.info(f"    {'recall_weighted':25s}: {metrics['recall_weighted']:.6f}")
    logger.info(f"    {'f1_weighted':25s}: {metrics['f1_weighted']:.6f}")

    logger.info("Per-class metrics:")
    for class_idx, scores in metrics["per_class"].items():
        label = class_labels[class_idx] if class_idx < len(class_labels) else str(class_idx)
        logger.info(
            f"    {label:12s}  P={scores['precision']:.4f}  "
            f"R={scores['recall']:.4f}  F1={scores['f1']:.4f}  n={scores['support']}"
        )

    logger.info("Inference timing:")
    logger.info(f"    {'samples':25s}: {timing['n_samples']}")
    logger.info(f"    {'inference_time_s':25s}: {timing['inference_time_s']:.4f}")
    logger.info(f"    {'ms_per_sample':25s}: {timing['ms_per_sample']:.4f}")
    logger.info(f"    {'samples_per_sec':25s}: {timing['samples_per_sec']:.2f}")


# =====================================================
# =========              Main                 =========
# =====================================================

def main(config):
    logger = config.get_logger("test")

    loader_type = config["data_loader"]["type"]
    loader_args = dict(config["data_loader"]["args"])
    loader_args.setdefault("batch_size", TEST_BATCH_SIZE)
    loader_args.setdefault("shuffle", False)
    loader_args.setdefault("num_workers", TEST_NUM_WORKERS)
    if hasattr(module_data, "_safe_num_workers"):
        loader_args["num_workers"] = module_data._safe_num_workers(
            loader_args["num_workers"]
        )
    loader_args["split"] = "test"

    data_loader = getattr(module_data, loader_type)(**loader_args)

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    # load checkpoint
    resume_path = config.resume
    if resume_path is None:
        raise RuntimeError(
            "No checkpoint given (DEFAULT_MODEL is unset — train first or pass -r path/to/model_best.pth)."
        )
    resume_path = Path(resume_path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # run inference (timed)
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    all_targets = []
    all_preds = []
    n_samples = len(data_loader.data_loader.dataset)

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_targets.append(target.detach().cpu())
            all_preds.append(preds.detach().cpu())

            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for m_idx, metric in enumerate(metric_fns):
                total_metrics[m_idx] += metric(output, target) * batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_s = time.perf_counter() - inference_start

    # loss + config metrics (accuracy, top_k_acc, ...)
    log = {"loss": total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    cm, n_classes = build_confusion_matrix(y_true, y_pred)
    class_labels = get_class_labels(n_classes)
    cls_metrics = module_metric.multiclass_metrics_from_confusion_matrix(cm)

    timing = {
        "n_samples": n_samples,
        "inference_time_s": inference_time_s,
        "ms_per_sample": (inference_time_s / n_samples) * 1000.0 if n_samples else 0.0,
        "samples_per_sec": n_samples / inference_time_s if inference_time_s > 0 else 0.0,
    }
    log_test_results(logger, cls_metrics, timing, class_labels)

    output_pdf = Path(config.log_dir) / "confusion_matrix.pdf"
    save_confusion_matrix(cm, n_classes, class_labels, output_pdf)
    logger.info(f"Saved confusion matrix to: {output_pdf}")


# =====================================================
# =========           Entry point             =========
# =====================================================

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH, type=str,
                      help="config file path (default: from params.py)")
    args.add_argument("-r", "--resume", default=DEFAULT_MODEL, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)
    main(config)
