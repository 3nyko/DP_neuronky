import argparse
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

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=True,
        # validation_split=0.0,
        # training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    resume_path = config.resume
    if resume_path is None:
        raise RuntimeError(
            "No checkpoint given (DEFAULT_MODEL is unset — train first or pass -r path/to/model_best.pth)."
        )
    resume_path = Path(resume_path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    logger.info('Loading checkpoint: {} ...'.format(resume_path))
    # Full checkpoints include pickled training state (not tensors-only); PyTorch 2.6+ defaults weights_only=True.
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_targets.append(target.detach().cpu())
            all_preds.append(preds.detach().cpu())

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.data_loader.dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # Build and save confusion matrix as PDF.
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    max_true = int(y_true.max()) if y_true.size else 0
    max_pred = int(y_pred.max()) if y_pred.size else 0
    n_classes = max(max_true, max_pred) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    class_labels = [str(i) for i in range(n_classes)]
    if n_classes == len(module_data.MULTICLASS_DICT):
        inv_map = {v: k for k, v in module_data.MULTICLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]
    elif n_classes == len(module_data.BINCLASS_DICT):
        inv_map = {v: k for k, v in module_data.BINCLASS_DICT.items()}
        class_labels = [inv_map.get(i, str(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

    fig.tight_layout()
    output_pdf = Path(config.log_dir) / "confusion_matrix.pdf"
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved confusion matrix to: {output_pdf}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=DEFAULT_CONFIG_PATH, type=str,
                      help='config file path (default: split)')
    args.add_argument('-r', '--resume', default=DEFAULT_MODEL, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
