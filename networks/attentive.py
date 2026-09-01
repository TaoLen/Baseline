import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from layers import AttentiveLayer, AttentiveFPAtomLayer
from activation import get_activation


class AttentiveNet(nn.Module):
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
        num_timesteps,
        num_tasks,
        mc_class_counts=None):

        super(AttentiveNet, self).__init__()
        self.num_tasks = num_tasks
        self.dropout_rate = dropout_rate
        self.num_timesteps = num_timesteps

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

        if len(agg_hidden_dims) == 0:
            raise ValueError(
                "agg_hidden_dims must contain at least one value")
        hidden_dim = agg_hidden_dims[0]
        if any(d != hidden_dim for d in agg_hidden_dims):
            raise ValueError(
                "AttentiveFP official uses a single hidden size. "
                "Provide equal values in agg_hidden_dims.")

        self.saved_embeddings = []
        self.hidden_dim = hidden_dim

        self.lin1 = nn.Linear(node_dim, hidden_dim)
        self.gate_conv = AttentiveLayer(
            hidden_dim,
            hidden_dim,
            edge_dim,
            dropout_rate
            )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.atom_convs = nn.ModuleList([
            AttentiveFPAtomLayer(
                hidden_dim, dropout_rate)
            for _ in range(max(num_agg_layers - 1, 0))
            ])
        self.atom_grus = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(max(num_agg_layers - 1, 0))
            ])

        self.mol_conv = AttentiveFPAtomLayer(
            hidden_dim, dropout_rate)
        self.mol_gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.agg_layers = [
            self.gate_conv, *self.atom_convs
            ]

        self.lin_layers = nn.ModuleList()
        for i in range(num_lin_layers):
            in_dim = hidden_dim if i == 0 else lin_hidden_dims[i - 1]
            out_dim = lin_hidden_dims[i]
            self.lin_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    get_activation(activation),
                    nn.Dropout(dropout_rate)
                )
            )
        if num_lin_layers > 0:
            self.embedding_dim = lin_hidden_dims[-1]
            self.embedding_layer = nn.Linear(
                lin_hidden_dims[-1],
                self.embedding_dim
            )
        else:
            self.embedding_dim = hidden_dim
            self.embedding_layer = nn.Identity()
        self.output_layer = nn.Linear(
            self.embedding_dim, num_tasks)
        self.mc_heads = nn.ModuleDict({
            str(i): nn.Linear(
                self.embedding_dim,
                int(self.mc_class_counts[i].item()))
            for i in self.mc_task_indices
            })

    def forward(
        self, data,
        save_embeddings=False,
        return_penultimate=False):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        x = F.leaky_relu(self.lin1(x))
        h = F.elu(self.gate_conv(
            x, edge_index, edge_attr))
        h = F.dropout(
            h,
            p=self.dropout_rate,
            training=self.training
            )
        x = self.gru(h, x).relu()

        for conv, gru in zip(
            self.atom_convs, self.atom_grus
            ):
            h = F.elu(conv(x, edge_index))
            h = F.dropout(
                h,
                p=self.dropout_rate,
                training=self.training
                )
            x = gru(h, x).relu()

        out = global_add_pool(x, batch).relu()
        row = torch.arange(
            batch.size(0), device=batch.device)
        mol_edge_index = torch.stack(
            [row, batch], dim=0)

        for _ in range(self.num_timesteps):
            h = F.elu(self.mol_conv(
                (x, out), mol_edge_index))
            h = F.dropout(
                h,
                p=self.dropout_rate,
                training=self.training
                )
            out = self.mol_gru(h, out).relu()

        out = F.dropout(
            out,
            p=self.dropout_rate,
            training=self.training
            )
        for lin in self.lin_layers:
            out = lin(out)
        embeddings = self.embedding_layer(out)
        penultimate = embeddings.clone()
        out = self.output_layer(embeddings)

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu()
            )
        if return_penultimate:
            return penultimate

        if not self.mc_heads:
            return out
        mc_logits = {}
        for idx, head in self.mc_heads.items():
            mc_logits[int(idx)] = head(embeddings)
        return {"scalar": out, "mc_logits": mc_logits}
