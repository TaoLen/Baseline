from initialization import (
    attentive_weights,
    gat_weights,
    gin_weights,
    mpnn_weights
    )
from utils import set_seed
         

def attentive_resets(model, seed=42):
    set_seed(seed)
    if hasattr(model, 'lin1') and hasattr(model.lin1, 'reset_parameters'):
        model.lin1.reset_parameters()
    if hasattr(model, 'gate_conv') and hasattr(model.gate_conv, 'reset_parameters'):
        model.gate_conv.reset_parameters()
    if hasattr(model, 'gru') and hasattr(model.gru, 'reset_parameters'):
        model.gru.reset_parameters()
    for conv in getattr(model, 'atom_convs', []):
        if hasattr(conv, 'reset_parameters'):
            conv.reset_parameters()
    for gru in getattr(model, 'atom_grus', []):
        if hasattr(gru, 'reset_parameters'):
            gru.reset_parameters()
    if hasattr(model, 'mol_conv') and hasattr(model.mol_conv, 'reset_parameters'):
        model.mol_conv.reset_parameters()
    if hasattr(model, 'mol_gru') and hasattr(model.mol_gru, 'reset_parameters'):
        model.mol_gru.reset_parameters()
    for lin_seq in getattr(model, 'lin_layers', []):
        attentive_weights(lin_seq)
    if hasattr(model, 'embedding_layer') and hasattr(model.embedding_layer, 'reset_parameters'):
        model.embedding_layer.reset_parameters()
    if hasattr(model, 'output_layer') and hasattr(model.output_layer, 'reset_parameters'):
        model.output_layer.reset_parameters()
    if hasattr(model, 'mc_heads'):
        for head in model.mc_heads.values():
            if hasattr(head, 'reset_parameters'):
                head.reset_parameters()


def gat_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        gat_weights(layer.conv)
    for layer_norm in model.norm_layers:
        gat_weights(layer_norm)
    for m in model.lin_layers:
        gat_weights(m)
    gat_weights(model.embedding_layer)
    gat_weights(model.output_layer)


def gin_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        gin_weights(layer.conv)
    for m in model.lin_layers:
        gin_weights(m)
    gin_weights(model.embedding_layer)
    gin_weights(model.output_layer)


def mpnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        mpnn_weights(layer)
    for lin_seq in model.lin_layers:
        mpnn_weights(lin_seq)
    mpnn_weights(model.embedding_layer)
    mpnn_weights(model.output_layer)
