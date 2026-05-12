import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "config_autoencoder_shallow.json")

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# =====================================================
# =========           Functions               =========
# =====================================================

def train_one_epoch(model, data_loader, optimizer, criterion, metric_fns, device):
    """Train one epoch, return dict with loss and metrics."""
    model.train()
    total_loss = 0.0
    total_metrics = {m.__name__: 0.0 for m in metric_fns}
    n_samples = 0

    for batch_data, _ in data_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_data)
        loss.backward()
        optimizer.step()

        bs = batch_data.size(0)
        total_loss += loss.item() * bs
        for m in metric_fns:
            total_metrics[m.__name__] += m(output, batch_data) * bs
        n_samples += bs

    result = {"loss": total_loss / n_samples}
    for k, v in total_metrics.items():
        result[k] = v / n_samples
    return result


def validate(model, data_loader, criterion, metric_fns, device):
    """Validate, return dict with loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_metrics = {m.__name__: 0.0 for m in metric_fns}
    n_samples = 0

    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_data)

            bs = batch_data.size(0)
            total_loss += loss.item() * bs
            for m in metric_fns:
                total_metrics[m.__name__] += m(output, batch_data) * bs
            n_samples += bs

    result = {"loss": total_loss / n_samples}
    for k, v in total_metrics.items():
        result[k] = v / n_samples
    return result


# =====================================================
# =========              Main                 =========
# =====================================================

def main(config):
    logger = config.get_logger("train")

    # Data
    data_loader_obj = config.init_obj("data_loader", module_data)
    train_loader = data_loader_obj.data_loader
    val_loader = data_loader_obj.valid_data_loader

    num_train = len(train_loader.dataset)
    num_val = len(val_loader.dataset) if val_loader else 0
    print(f"\nDataset info:")
    print(f"\tTraining samples:   {num_train} ({len(train_loader)} batches)")
    print(f"\tValidation samples: {num_val} ({len(val_loader) if val_loader else 0} batches)")

    # Model
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU available: True | using device: {device} ({gpu_name})")
    else:
        logger.info("GPU available: False | using CPU")
    model = model.to(device)

    # Loss, metrics, optimizer, scheduler
    criterion = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # Training loop
    epochs = config["trainer"]["epochs"]
    early_stop = config["trainer"].get("early_stop", 10)
    save_dir = Path(config.save_dir)

    best_val_loss = float("inf")
    best_val_re_pct = float("inf")
    no_improve_count = 0

    train_start = datetime.now()
    logger.info(f"Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, metric_fns, device)

        val_result = None
        if val_loader:
            val_result = validate(model, val_loader, criterion, metric_fns, device)

        lr_scheduler.step()

        # Log in the same format as regular trainer
        logger.info(f"    {'epoch':15s}: {epoch}")
        for key, value in train_result.items():
            logger.info(f"    {key:15s}: {value}")
        if val_result:
            for key, value in val_result.items():
                logger.info(f"    {'val_' + key:15s}: {value}")

        # Save best
        check_loss = val_result["loss"] if val_result else train_result["loss"]
        if check_loss < best_val_loss:
            best_val_loss = check_loss
            if val_result and "reconstruction_error" in val_result:
                best_val_re_pct = val_result["reconstruction_error"] / 255.0 * 100.0
            no_improve_count = 0
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "config": config.config,
            }
            best_path = save_dir / "model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saving current best: model_best.pth ...")
        else:
            no_improve_count += 1

        # Print best so far at end of each epoch
        logger.info(f"    {'best_val_RE%':15s}: {best_val_re_pct:.4f}%")

        if no_improve_count >= early_stop:
            logger.info(f"Validation performance didn't improve for {early_stop} epochs. Training stops.")
            break

    train_end = datetime.now()
    elapsed = train_end - train_start
    logger.info(f"Training finished at {train_end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {elapsed.total_seconds():.2f}s ({elapsed})")


# =====================================================
# =========           Entry point             =========
# =====================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train autoencoder")
    ap.add_argument("-c", "--config", default=DEFAULT_CONFIG, type=str,
                    help="config file path")
    ap.add_argument("-d", "--device", default=None, type=str,
                    help="indices of GPUs to enable")
    args = ap.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = ConfigParser(config=config_dict, resume=None, modification=None)
    main(config)
