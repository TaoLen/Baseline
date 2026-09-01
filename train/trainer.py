import copy
import torch
import optuna
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils import (
    device, 
    clip_gradients
    )

from loss import (
    MaskedLoss,
    compute_loss_matrix,
    reduce_task_mean
    )

from save import save_embeddings


def get_loss(
    num_tasks,
    task_type=None,
    mc_label_values=None):

    loss_fn = MaskedLoss(
        num_tasks=num_tasks,
        task_type=task_type,
        mc_label_values=mc_label_values,
        )
    loss_fn._task_type = task_type
    loss_fn._mc_label_values = mc_label_values
    return loss_fn



def _loss_stats(y_pred, y_true, loss_fn):
    task_type = getattr(loss_fn, "_task_type", None)
    mc_label_values = getattr(loss_fn, "_mc_label_values", None)
    loss_mat, mask, task_type = compute_loss_matrix(
        y_pred, y_true,
        task_type=task_type,
        mc_label_values=mc_label_values
        )

    task_means, valid = reduce_task_mean(
        loss_mat, mask.float())

    is_bin = (task_type == 1)
    is_mc = (task_type == 2)
    is_reg = (task_type == 0)

    def sum_count(type_mask):
        valid_type = valid & type_mask
        return task_means[valid_type].sum(), valid_type.sum()

    total_sum = task_means[valid].sum()
    total_count = valid.sum()
    bin_sum, bin_count = sum_count(is_bin)
    mc_sum, mc_count = sum_count(is_mc)
    reg_sum, reg_count = sum_count(is_reg)

    return {
        "total_sum": total_sum,
        "total_count": total_count,
        "bin_sum": bin_sum,
        "bin_count": bin_count,
        "mc_sum": mc_sum,
        "mc_count": mc_count,
        "reg_sum": reg_sum,
        "reg_count": reg_count,
    }


def train_epoch(
    model, 
    optimizer, 
    data_loader, 
    loss_fn, 
    max_grad_norm=1.0, 
    clip_method='norm',
    return_stats=False):

    model.train()
    total_loss = 0.0
    total_count = 0.0
    bin_sum = 0.0
    bin_count = 0.0
    mc_sum = 0.0
    mc_count = 0.0
    reg_sum = 0.0
    reg_count = 0.0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        if isinstance(data, Batch):
            data = data.to(device)
            labels = data.y.to(device
                ) if hasattr(data, 'y') else None
            out = model(data)
            loss = loss_fn(out, labels)
        else:
            continue

        loss.backward()
        clip_gradients(model, 
            max_grad_norm, 
            method=clip_method
            )
        optimizer.step()
        stats = _loss_stats(out, labels, loss_fn)
        total_loss += float(stats["total_sum"].item())
        total_count += float(stats["total_count"].item())
        bin_sum += float(stats["bin_sum"].item())
        bin_count += float(stats["bin_count"].item())
        mc_sum += float(stats["mc_sum"].item())
        mc_count += float(stats["mc_count"].item())
        reg_sum += float(stats["reg_sum"].item())
        reg_count += float(stats["reg_count"].item())

    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    if not return_stats:
        return avg_loss
    stats = {
        "bin": bin_sum / bin_count if bin_count > 0 else float("nan"),
        "mc": mc_sum / mc_count if mc_count > 0 else float("nan"),
        "reg": reg_sum / reg_count if reg_count > 0 else float("nan"),
        "bin_count": bin_count,
        "mc_count": mc_count,
        "reg_count": reg_count,
        "total_count": total_count,
    }
    return avg_loss, stats


def evaluate(
    model, 
    data_loader, 
    loss_fn,
    return_stats=False):

    model.eval()
    total_loss = 0.0
    total_count = 0.0
    bin_sum = 0.0
    bin_count = 0.0
    mc_sum = 0.0
    mc_count = 0.0
    reg_sum = 0.0
    reg_count = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if data is None:
                continue

            if isinstance(data, Batch): 
                data = data.to(device)
                labels = data.y.to(device
                    ) if hasattr(data, 'y') else None
                out = model(data)
            else:
                continue

            stats = _loss_stats(out, labels, loss_fn)
            total_loss += float(stats["total_sum"].item())
            total_count += float(stats["total_count"].item())
            bin_sum += float(stats["bin_sum"].item())
            bin_count += float(stats["bin_count"].item())
            mc_sum += float(stats["mc_sum"].item())
            mc_count += float(stats["mc_count"].item())
            reg_sum += float(stats["reg_sum"].item())
            reg_count += float(stats["reg_count"].item())

    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    if not return_stats:
        return avg_loss
    stats = {
        "bin": bin_sum / bin_count if bin_count > 0 else float("nan"),
        "mc": mc_sum / mc_count if mc_count > 0 else float("nan"),
        "reg": reg_sum / reg_count if reg_count > 0 else float("nan"),
        "bin_count": bin_count,
        "mc_count": mc_count,
        "reg_count": reg_count,
        "total_count": total_count,
    }
    return avg_loss, stats


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    patience,
    delta,
    window_size,
    best_model=True,
    warm_up_epochs=3,
    eta_min=0.001,
    enable_pruning=True):

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=warm_up_epochs,
        min_lr=eta_min
        )
    best_val_loss = float('inf')
    min_val_loss = float('inf')
    best_model_state = None
    best_epoch = None
    epochs_no_improve = 0
    val_loss_window = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn
            )
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate(model, val_loader, loss_fn)
        val_losses.append(avg_val_loss)
        val_loss_window.append(avg_val_loss)
        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)
        avg_val_loss_window = (sum(val_loss_window)
            / len(val_loss_window)
            )
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_train_loss:.4f} - "
            f"Val: {avg_val_loss:.4f} - "
            f"Win: {avg_val_loss_window:.4f}"
            )
        if (avg_val_loss_window
            < best_val_loss - delta):

            best_val_loss = avg_val_loss_window
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(
                model.state_dict()
                )
            best_epoch = epoch + 1
        if best_model:
            save_embeddings(
                model,
                train_loader,
                epoch,
                "../output/embeddings"
                )
        if (enable_pruning
            and val_losses
            and val_losses[0] < 0.60):
            print(f"Pruned: low initial loss: "
                f"{val_losses[0]:.4f}"
                )
            raise optuna.exceptions.TrialPruned()
        scheduler.step(avg_val_loss)

    if best_model_state is not None:
        print(f"Restoring best model from epoch "
            f"{best_epoch} with val_loss "
            f"{min_val_loss:.4f}"
            )
        model.load_state_dict(best_model_state)

    return (
        best_val_loss,
        min_val_loss,
        train_losses,
        val_losses
        )
