import torch.optim as optim

from attentive import AttentiveNet
from gat import GATNet
from gin import GINet
from mpnn import MPNNet

def configure_optimizer(trial, model, lr):
    
    optimizer_name = trial.suggest_categorical(
        'optimizer', [
             'RAdam', 'Adam', 'RMSprop', 'SGD'])
    weight_decay = trial.suggest_float(
        'weight_decay', 1e-8, 1e-4)
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
        )
    return optimizer


def configure_attentive(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,
    mc_class_counts=None):
    
    num_agg_layers = trial.suggest_int(
        'num_agg_layers', 2, 6)
    first_dim = trial.suggest_int(
        'agg_hidden_dim_1', 10, 500)
    agg_hidden_dims = [first_dim]
    for i in range(1, num_agg_layers):
        agg_hidden_dims.append(
            trial.suggest_int(
                f'agg_hidden_dim_{i+1}',
                first_dim, first_dim)
            )

    num_lin_layers = trial.suggest_int(
        'num_lin_layers', 2, 4)
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(num_lin_layers)
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    num_timesteps = trial.suggest_int(
        'num_timesteps', 1, 3
        )
    
    model = AttentiveNet(
        node_dim, 
        edge_dim, 
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate,
        num_timesteps, 
        num_tasks,
        mc_class_counts=mc_class_counts
        )
    
    return model



def configure_gat(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,
    mc_class_counts=None):

    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i + 1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i + 1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    heads = trial.suggest_int(
        'heads', 1, 12
        )

    model = GATNet(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        heads,
        num_tasks,
        mc_class_counts=mc_class_counts
        )

    return model


def configure_gin(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,
    mc_class_counts=None):

    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i+1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )

    model = GINet(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        num_tasks,
        mc_class_counts=mc_class_counts
        )

    return model


def configure_mpnn(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,
    mc_class_counts=None):
    
    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    
    model = MPNNet(
        node_dim, 
        edge_dim, 
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate, 
        num_tasks,
        mc_class_counts=mc_class_counts
        )
    
    return model
