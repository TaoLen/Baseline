import torch.optim as optim

from utils import device, set_seed
from attentive import AttentiveNet
from gat import GATNet
from gin import GINet
from mpnn import MPNNet

from hyperparams import (
    configure_optimizer,
    configure_attentive,
    configure_gat,
    configure_gin,
    configure_mpnn,
    ) 
from resets import (
    attentive_resets,
    gat_resets,
    gin_resets,
    mpnn_resets,
    )
from trainer import (
    get_loss, 
    train_model
    )
from loaders import infer_task_metadata


def objective(
    trial, 
    node_dim, 
    edge_dim, 
    train_loader, 
    val_loader, 
    num_tasks=None, 
    architecture_type='mpnn',
    lr=None,
    task_type=None,
    mc_class_counts=None,
    mc_label_values=None):

    set_seed(42)
    if task_type is None or mc_class_counts is None or mc_label_values is None:
        inferred_type, inferred_counts, inferred_labels = (
            infer_task_metadata(train_loader)
            )
        if task_type is None:
            task_type = inferred_type
        if mc_class_counts is None:
            mc_class_counts = inferred_counts
        if mc_label_values is None:
            mc_label_values = inferred_labels

    if architecture_type == 'attentive':
        model = configure_attentive(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'gat':
        model = configure_gat(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'gin':
        model = configure_gin(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'mpnn':
        model = configure_mpnn(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    else:
        raise ValueError(
            f"Invalid architecture: {architecture_type}")

    model.to(device)
    model.task_type = task_type
    model.mc_label_values = mc_label_values

    if architecture_type == 'attentive':
        attentive_resets(model)
    elif architecture_type == 'gat':
        gat_resets(model)
    elif architecture_type == 'gin':
        gin_resets(model)
    elif architecture_type == 'mpnn':
        mpnn_resets(model)

    loss_fn = get_loss(
        num_tasks, 
        task_type=task_type,
        mc_label_values=mc_label_values
        )
    optimizer = configure_optimizer(
        trial, model, lr=lr
        )
    _, min_val_loss, _, _ = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_fn, 
        num_epochs=2000,  
        patience=5, 
        delta=0.01, 
        window_size=5, 
        best_model=False,
        enable_pruning=True
        )
    
    return min_val_loss


def retrain(
    best_params, 
    node_dim, 
    edge_dim, 
    train_loader, 
    val_loader, 
    num_tasks,
    architecture_type='mpnn',
    lr=None,
    task_type=None,
    mc_class_counts=None,
    mc_label_values=None):

    set_seed(42)
    agg_hidden_dims = [
        best_params[f'agg_hidden_dim_{i+1}'] 
        for i in range(
            best_params['num_agg_layers'])
            ]
    lin_hidden_dims = [
        best_params[f'lin_hidden_dim_{i+1}'] 
        for i in range(
            best_params['num_lin_layers'])
            ] 
    if task_type is None or mc_class_counts is None or mc_label_values is None:
        inferred_type, inferred_counts, inferred_labels = (
            infer_task_metadata(train_loader)
            )
        if task_type is None:
            task_type = inferred_type
        if mc_class_counts is None:
            mc_class_counts = inferred_counts
        if mc_label_values is None:
            mc_label_values = inferred_labels

    if architecture_type == 'attentive':
        model = AttentiveNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'], 
            best_params['num_timesteps'],
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'gat':
        model = GATNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'],
            best_params['heads'], 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'gin':
        model = GINet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'],
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    elif architecture_type == 'mpnn':
        model = MPNNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'], 
            num_tasks,
            mc_class_counts=mc_class_counts
            )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture_type}"
            )
    
    model.to(device)
    model.task_type = task_type
    model.mc_label_values = mc_label_values

    if architecture_type == 'attentive':
        attentive_resets(model)
    elif architecture_type == 'gat':
        gat_resets(model)
    elif architecture_type == 'gin':
        gin_resets(model)
    elif architecture_type == 'mpnn':
        mpnn_resets(model)

    optimizer = getattr(optim, best_params['optimizer'])(
        model.parameters(), lr=lr, 
        weight_decay=best_params['weight_decay']
        )
    loss_fn = get_loss(
        num_tasks, 
        task_type=task_type,
        mc_label_values=mc_label_values
        ) 
    best_val_loss, min_val_loss, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_fn, 
        num_epochs=2000,  
        patience=10, 
        delta=0.01, 
        window_size=5, 
        best_model=True,
        enable_pruning=False
        )

    return model, best_val_loss, min_val_loss,  train_losses, val_losses
