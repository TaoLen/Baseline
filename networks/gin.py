import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool
    )

from layers import GINLayer
from activation import get_activation


class GINet(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation,
        dropout_rate,
        num_tasks,
        mc_class_counts=None):

        super(GINet, self).__init__()
        self.num_tasks = num_tasks

        if mc_class_counts is None:
            mc_class_counts = torch.zeros(
                num_tasks, dtype=torch.long)
        elif not torch.is_tensor(mc_class_counts):
            mc_class_counts = torch.tensor(
                mc_class_counts, dtype=torch.long)
        else:
            mc_class_counts = mc_class_counts.to(
                torch.long)
        if mc_class_counts.numel() != num_tasks:
            raise ValueError(
                "mc_class_counts must have length num_tasks")
        self.register_buffer(
            "mc_class_counts", mc_class_counts)
        self.mc_task_indices = [
            i for i in range(num_tasks)
            if int(self.mc_class_counts[i].item()) > 1
            ]
        self._agg_out_dim = agg_hidden_dims[-1]
        input_dims = [node_dim] + agg_hidden_dims[:-1]
        self.agg_layers = nn.ModuleList([
            GINLayer(
                input_dims[i],
                agg_hidden_dims[i],
                edge_dim,
                num_lin_layers)
            for i in range(num_agg_layers)])
        self.lin_layers = nn.ModuleList()
        for i, out_dim in enumerate(lin_hidden_dims):
            in_dim = (self._agg_out_dim
                if i == 0 else lin_hidden_dims[i - 1])
            self.lin_layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)]
        self.embedding_dim = lin_hidden_dims[-1]
        self.embedding_layer = nn.Linear(
            lin_hidden_dims[-1],
            self.embedding_dim)
        self.output_layer = nn.Linear(
            self.embedding_dim,
            num_tasks)
        self.mc_heads = nn.ModuleDict({
            str(i): nn.Linear(
                self.embedding_dim,
                int(self.mc_class_counts[i].item()))
            for i in self.mc_task_indices
            })

    def forward(
        self,
        data,
        save_embeddings=False,
        return_penultimate=False):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch)
        for layer in self.agg_layers:
            x = layer(x, edge_index, edge_attr, batch)
        x = global_add_pool(x, batch)
        for layer in self.lin_layers:
            x = layer(x)
        embeddings = self.embedding_layer(x)
        penultimate = embeddings.clone()
        out = self.output_layer(embeddings)
        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu())
            
        if return_penultimate:
            return penultimate
        if not self.mc_heads:
            return out
        mc_logits = {}
        for idx, head in self.mc_heads.items():
            mc_logits[int(idx)] = head(embeddings)
        return {"scalar": out, "mc_logits": mc_logits}
