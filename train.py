import argparse
import collections
import os
import sys
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from params import DEFAULT_CONFIG_PATH

# =====================================================
# =========       Constants and options       =========
# =====================================================

SEED = 123

TENSORBOARD_PORT = 6006
TENSORBOARD_HOST = "127.0.0.1"
TENSORBOARD_RELOAD_INTERVAL = 10
TENSORBOARD_KEEP_LAST = 15

CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
CLI_OPTIONS = [
    CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
    CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
]

# =====================================================
# =========        Reproducibility            =========
# =====================================================

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# =====================================================
# =========           Functions               =========
# =====================================================

def print_data_info(data_loader, valid_data_loader):
    """Print train and val sample/batch counts."""
    num_train_samples = len(data_loader.dataset) if hasattr(data_loader, "dataset") else len(data_loader.data_loader.dataset)
    num_val_samples = len(valid_data_loader.dataset) if valid_data_loader is not None else 0

    num_train_batches = len(data_loader)
    num_val_batches = len(valid_data_loader) if valid_data_loader is not None else 0

    print("\nDataset info:")
    print(f"\tTraining samples:   {num_train_samples} ({num_train_batches} batches)")
    print(f"\tValidation samples: {num_val_samples} ({num_val_batches} batches)")


def _latest_event_mtime(run_dir: Path) -> float:
    """Return mtime of newest TensorBoard event file inside run_dir, or 0 if none."""
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return 0.0
    return max(f.stat().st_mtime for f in event_files)


def _get_latest_run_by_event(base: Path) -> Path | None:
    """Pick the run dir whose newest event file was modified most recently."""
    best_dir = None
    best_time = 0.0

    for d in base.iterdir():
        if not d.is_dir():
            continue
        t = _latest_event_mtime(d)
        if t > best_time:
            best_time = t
            best_dir = d

    return best_dir


def _sanitize_tb_label(label: str) -> str:
    """Remove characters that break TensorBoard logdir_spec."""
    return label.replace(":", "_").replace(",", "_")


def start_tensorboard_for_latest(
    base_log_dir="saved/log/NN_1",
    port=TENSORBOARD_PORT,
    keep_last=TENSORBOARD_KEEP_LAST,
    reload_interval=TENSORBOARD_RELOAD_INTERVAL,
    current_run_dir: Path | None = None,
    current_run_label: str | None = None,
):
    """Launch TensorBoard showing the current run + recent history."""
    base = Path(base_log_dir)
    if not base.exists():
        print(f"[TensorBoard] Log dir not found: {base.resolve()}")
        return

    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[TensorBoard] No subdirectories in {base}")
        return

    latest_run = current_run_dir if current_run_dir is not None else _get_latest_run_by_event(base)
    if latest_run is None:
        print(f"[TensorBoard] No event files found in {base} (events.out.tfevents.*).")
        return

    subdirs_sorted_by_event = sorted(subdirs, key=_latest_event_mtime)
    recent_runs = [d for d in subdirs_sorted_by_event if _latest_event_mtime(d) > 0][-keep_last:]
    recent_runs = [d for d in recent_runs if d.resolve() != latest_run.resolve()]

    parts = [f"{d.name}:{d}" for d in recent_runs]
    active_label = _sanitize_tb_label(current_run_label or latest_run.name)
    parts.append(f"{active_label}:{latest_run}")
    logdir_spec = ",".join(parts)

    print(f"[TensorBoard] Base: {base}")
    print(f"[TensorBoard] Active run: {active_label}")
    print(f"[TensorBoard] Active run dir: {latest_run.name}")
    print(f"[TensorBoard] Showing {len(recent_runs)} previous runs + active run")
    print(f"[TensorBoard] Reload interval: {reload_interval}s")

    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir_spec", logdir_spec,
        "--reload_interval", str(reload_interval),
        "--port", str(port),
        "--host", TENSORBOARD_HOST,
    ]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()


# =====================================================
# =========              Main                 =========
# =====================================================

def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()
    print_data_info(data_loader, valid_data_loader)

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device.index or 0)
        logger.info(f"GPU available: True | using device: {device} ({gpu_name})")
    else:
        logger.info("GPU available: False | using CPU")
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer and learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    # launch TensorBoard
    run_id = Path(config.log_dir).name
    experiment_name = config["name"]
    current_run_label = f"{experiment_name}_{run_id}"
    start_tensorboard_for_latest(
        base_log_dir=str(Path(config.log_dir).parent),
        current_run_dir=Path(config.log_dir),
        current_run_label=current_run_label,
    )

    # train
    train_start = datetime.now()
    logger.info(f"Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}")

    trainer.train()

    train_end = datetime.now()
    elapsed = train_end - train_start
    logger.info(f"Training finished at {train_end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {elapsed.total_seconds():.2f}s ({elapsed})")


# =====================================================
# =========           Entry point             =========
# =====================================================

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH, type=str,
                      help="config file path (default: from params.py)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args, CLI_OPTIONS)
    main(config)
