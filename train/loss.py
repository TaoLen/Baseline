import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from utils import task_inference


def build_mc_label_maps(mc_label_values):
    if mc_label_values is None:
        return None
    maps = []
    for vals in mc_label_values:
        if vals is None:
            maps.append(None)
            continue
        maps.append({int(v): i for i, v in enumerate(vals)})
    return maps


def prepare_masked_data(
    y_true, y_pred):

    if isinstance(y_true, (Batch, Data)):
        y_true = y_true.y
    if isinstance(y_pred, (Batch, Data)):
        y_pred = y_pred.y
    if not torch.is_tensor(y_true):
        raise ValueError("y_true must be a tensor")
    if not torch.is_tensor(y_pred):
        raise ValueError("y_pred must be a tensor")

    if y_true.dim() != 2:
        raise ValueError("y_true must be 2D [batch, num_tasks]")

    if y_pred.dim() == 2:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must match shape"
            )
    elif y_pred.dim() == 3:
        if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
            raise ValueError(
                "y_true and y_pred must match in first two dims"
            )
    else:
        raise ValueError("y_pred must be 2D or 3D")

    mask = ~torch.isnan(y_true)

    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)

    return y_true, y_pred, mask


def _split_pred(y_pred):
    if isinstance(y_pred, dict):
        pred_scalar = y_pred.get("scalar")
        pred_logits = y_pred.get("mc_logits", {})
    elif y_pred.dim() == 2:
        pred_scalar = y_pred
        pred_logits = None
    elif y_pred.dim() == 3:
        pred_scalar = y_pred[..., 0]
        pred_logits = y_pred
    else:
        raise ValueError("y_pred must be 2D, 3D, or dict")
    if pred_scalar is None:
        raise ValueError("y_pred must contain scalar logits")
    return pred_scalar, pred_logits


def map_multiclass_targets(targets, mapping, num_classes):
    targets = targets.to(torch.long)
    if mapping is None:
        raise ValueError("Missing label mapping for multiclass task.")

    mapped = []
    for v in targets.tolist():
        if int(v) not in mapping:
            if int(v) < 0 or int(v) >= num_classes:
                raise ValueError(
                    "Unseen multiclass label encountered.")
            mapped.append(int(v))
        else:
            mapped.append(mapping[int(v)])
    return torch.tensor(
        mapped, device=targets.device, dtype=torch.long)


def reduce_task_mean(loss_mat, mask):
    task_counts = mask.sum(dim=0)
    task_sums = (loss_mat * mask).sum(dim=0)
    valid = task_counts > 0
    task_means = torch.zeros_like(task_sums)
    task_means[valid] = task_sums[valid] / task_counts[valid]
    return task_means, valid


def compute_loss_matrix(
    y_pred,
    y_true,
    task_type=None,
    mc_label_values=None,
    mc_label_maps=None):

    pred_scalar, pred_logits = _split_pred(y_pred)
    y_true, y_pred_scalar, mask = prepare_masked_data(
        y_true, pred_scalar)

    if task_type is None:
        task_type = task_inference(
            y_true, mask).to(y_true.device)
    else:
        task_type = task_type.to(y_true.device)

    if mc_label_maps is None:
        mc_label_maps = build_mc_label_maps(mc_label_values)

    loss_cls = F.binary_cross_entropy_with_logits(
        y_pred_scalar, y_true, reduction='none')
    loss_reg = F.smooth_l1_loss(
        y_pred_scalar, y_true, beta=0.5, reduction='none')

    is_bin = (task_type == 1).to(y_pred_scalar.device)
    is_mc = (task_type == 2).to(y_pred_scalar.device)

    loss = torch.where(is_bin.view(
        1, -1), loss_cls, loss_reg)

    if is_mc.any():
        if pred_logits is None:
            raise ValueError(
                "Multiclass task detected but logits are missing. "
                "Model must output per-task multiclass logits."
            )

        B, T = y_pred_scalar.shape
        loss_mc = torch.zeros(
            (B, T), device=y_pred_scalar.device,
            dtype=y_pred_scalar.dtype)

        for j in torch.where(is_mc)[0].tolist():
            valid = mask[:, j]
            if not valid.any():
                continue

            targets = y_true[valid, j].round().to(torch.long)
            mapping = None
            if mc_label_maps is not None:
                mapping = mc_label_maps[j]
            if mapping is None:
                raise ValueError(
                    "Missing label mapping for multiclass task.")

            if isinstance(pred_logits, dict):
                logits = pred_logits.get(j)
            else:
                logits = pred_logits[:, j, :]
            if logits is None:
                raise ValueError(
                    "Missing logits for multiclass task.")
            if logits.size(1) != len(mapping):
                raise ValueError(
                    "Multiclass logits size does not match label mapping.")
            mapped = map_multiclass_targets(
                targets, mapping, logits.size(1))
            l = F.cross_entropy(logits[valid], mapped, reduction='none')

            tmp = torch.zeros(
                B, device=y_pred_scalar.device,
                dtype=y_pred_scalar.dtype)
            tmp[valid] = l
            loss_mc[:, j] = tmp

        loss = torch.where(is_mc.view(1, -1), loss_mc, loss)

    return loss, mask, task_type


def MaskedLoss(
    num_tasks=None, task_type=None, mc_label_values=None):
    if num_tasks is None:
        raise ValueError("num_tasks must be provided")

    mc_label_maps = build_mc_label_maps(mc_label_values)

    def masked_loss_function(y_pred, y_true):
        if isinstance(y_pred, dict):
            y_pred_scalar = y_pred.get("scalar")
            y_pred_logits = y_pred.get("mc_logits", {})
        else:
            y_pred_scalar = y_pred
            y_pred_logits = None
        y_true, y_pred, mask = prepare_masked_data(
            y_true, y_pred_scalar)

        tt = task_type
        if tt is None:
            tt = task_inference(
                y_true, mask).to(y_true.device)
        else:
            tt = tt.to(y_true.device)

        if y_pred_scalar is None:
            raise ValueError("y_pred must contain scalar logits")

        loss_cls = F.binary_cross_entropy_with_logits(
            y_pred_scalar, y_true, reduction='none')
        loss_reg = F.smooth_l1_loss(
            y_pred_scalar, y_true, beta=0.5, reduction='none')

        is_bin = (tt == 1).to(y_pred_scalar.device)
        is_mc = (tt == 2).to(y_pred_scalar.device)

        loss = torch.where(is_bin.view(
            1, -1), loss_cls, loss_reg)

        if is_mc.any():
            if y_pred_logits is None:
                raise ValueError(
                    "Multiclass task detected but logits are missing. "
                    "Model must output per-task multiclass logits."
                )

            B, T = y_pred_scalar.shape
            loss_mc = torch.zeros(
                (B, T), device=y_pred_scalar.device,
                dtype=y_pred_scalar.dtype)

            for j in torch.where(is_mc)[0].tolist():
                valid = mask[:, j]
                if not valid.any():
                    continue

                targets = y_true[valid, j].round().to(torch.long)
                mapping = None
                if mc_label_maps is not None:
                    mapping = mc_label_maps[j]
                if mapping is None:
                    raise ValueError(
                        "Missing label mapping for multiclass task.")

                if isinstance(y_pred_logits, dict):
                    logits = y_pred_logits.get(j)
                else:
                    logits = y_pred_logits[:, j, :]
                if logits is None:
                    raise ValueError(
                        "Missing logits for multiclass task.")
                if logits.size(1) != len(mapping):
                    raise ValueError(
                        "Multiclass logits size does not match label mapping.")
                mapped = map_multiclass_targets(
                    targets, mapping, logits.size(1))
                l = F.cross_entropy(logits[valid], mapped, reduction='none')

                tmp = torch.zeros(
                    B, device=y_pred_scalar.device,
                    dtype=y_pred_scalar.dtype)
                tmp[valid] = l
                loss_mc[:, j] = tmp

            loss = torch.where(is_mc.view(1, -1), loss_mc, loss)
        
        task_means, valid = reduce_task_mean(
            loss, mask.float())
        if not valid.any():
            return torch.tensor(
                0.0, device=loss.device, requires_grad=True)
        return task_means[valid].mean()

    return masked_loss_function
