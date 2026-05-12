import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "config_autoencoder_shallow.json")

TEST_BATCH_SIZE = 512
TEST_NUM_WORKERS = 2


# =====================================================
# =========           Functions               =========
# =====================================================

def compute_reconstruction_errors(model, data_loader, device):
    """Compute per-sample MSE reconstruction error."""
    model.eval()
    errors = []
    labels = []

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(data_loader):
            batch_data = batch_data.to(device)
            output = model(batch_data)
            mse_per_sample = torch.mean((output - batch_data) ** 2, dim=1)
            errors.append(mse_per_sample.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(errors), np.concatenate(labels)


def find_best_threshold(errors, labels):
    """Find threshold that maximizes accuracy (binary: 0=benign, 1=attack)."""
    thresholds = np.linspace(errors.min(), np.percentile(errors, 99), 1000)
    best_acc = 0.0
    best_thresh = 0.0

    for t in thresholds:
        preds = (errors > t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh, best_acc


def save_error_histogram(errors, labels, threshold, output_path):
    """Save histogram of reconstruction errors (benign vs attack)."""
    benign_errors = errors[labels == 0]
    attack_errors = errors[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(benign_errors, bins=100, alpha=0.6, label="BENIGN", color="green")
    ax.hist(attack_errors, bins=100, alpha=0.6, label="ATTACK", color="red")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================
# =========              Main                 =========
# =====================================================

def main(config):
    logger = config.get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU available: True | using device: {device} ({gpu_name})")
    else:
        logger.info("GPU available: False | using CPU")

    # Load test data (contains both benign and attacks)
    data_dir = config["data_loader"]["args"]["data_dir"]
    mode = config["data_loader"]["args"].get("mode", "hexadecimal")
    test_dataset = module_data.CICIoV2024_Autoencoder_Dataset(data_dir=data_dir, mode=mode, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=TEST_NUM_WORKERS
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Build model
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Load checkpoint
    resume_path = config.resume
    if resume_path is None:
        raise RuntimeError(
            "No checkpoint given. Pass -r path/to/model_best.pth"
        )
    resume_path = Path(resume_path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    logger.info(f"Loading checkpoint: {resume_path} ...")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Compute reconstruction errors
    errors, labels = compute_reconstruction_errors(model, test_loader, device)

    benign_count = np.sum(labels == 0)
    attack_count = np.sum(labels == 1)
    logger.info(f"    {'benign_samples':15s}: {benign_count}")
    logger.info(f"    {'attack_samples':15s}: {attack_count}")
    logger.info(f"    {'mean_err_benign':15s}: {errors[labels == 0].mean():.6f}")
    logger.info(f"    {'mean_err_attack':15s}: {errors[labels == 1].mean():.6f}")

    # Find best threshold
    threshold, best_acc = find_best_threshold(errors, labels)

    # Compute metrics at threshold
    preds = (errors > threshold).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f"    {'threshold':15s}: {threshold:.6f}")
    logger.info(f"    {'accuracy':15s}: {accuracy:.6f}")
    logger.info(f"    {'precision':15s}: {precision:.6f}")
    logger.info(f"    {'recall':15s}: {recall:.6f}")
    logger.info(f"    {'f1_score':15s}: {f1:.6f}")
    logger.info(f"    {'TP':15s}: {tp}")
    logger.info(f"    {'TN':15s}: {tn}")
    logger.info(f"    {'FP':15s}: {fp}")
    logger.info(f"    {'FN':15s}: {fn}")

    # Save histogram
    output_pdf = Path(config.log_dir) / "reconstruction_error_histogram.pdf"
    save_error_histogram(errors, labels, threshold, output_pdf)
    logger.info(f"Saved histogram to: {output_pdf}")


# =====================================================
# =========           Entry point             =========
# =====================================================

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test autoencoder anomaly detection")
    args.add_argument("-c", "--config", default=DEFAULT_CONFIG, type=str,
                      help="config file path")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to model_best.pth checkpoint")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable")

    config = ConfigParser.from_args(args)
    main(config)
