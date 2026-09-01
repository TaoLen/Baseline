import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    softmax
    )
from torch_geometric.nn import (
    MessagePassing, 
    GATConv, 
    GATv2Conv, 
    GINConv,
    GraphNorm
    )
from torch_geometric.nn.inits import (
    glorot, 
    zeros
    )


class AttentiveLayer(MessagePassing):

    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate):

        super().__init__(aggr='add', node_dim=0)
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.edge_dim = edge_dim
        self.dropout = dropout_rate
        self.lin1 = nn.Linear(
            input_dim + edge_dim,
            output_dim,
            bias=False
            )
        self.lin2 = nn.Linear(
            output_dim,
            output_dim,
            bias=False
            )
        self.att_l = nn.Parameter(
            torch.empty(1, output_dim)
            )
        self.att_r = nn.Parameter(
            torch.empty(1, input_dim)
            )
        self.bias = nn.Parameter(
            torch.empty(output_dim)
            )
        self.last_attn_weights = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, batch=None):
        alpha = self.edge_updater(
            edge_index,
            x=x,
            edge_attr=edge_attr
            )
        out = self.propagate(
            edge_index,
            x=x,
            alpha=alpha
            )
        out = out + self.bias
        return out

    def edge_update(self, x_j, x_i, edge_attr,
        index, ptr, size_i):

        x_j = torch.cat([x_j, edge_attr], dim=-1)
        x_j = self.lin1(x_j)
        x_j = F.leaky_relu_(x_j)
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training
            )
        self.last_attn_weights = alpha
        return alpha

    def message(self, x_j, alpha):
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFPAtomLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.conv = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout_rate,
            add_self_loops=False,
            negative_slope=0.01
            )
        self.last_attn_weights = None

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out, (ei, aw) = self.conv(
            x, edge_index,
            return_attention_weights=True
            )
        self.last_attn_weights = aw.mean(dim=-1)
        return out


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        heads,
        dropout,
        concat=True):

        super(GATLayer, self).__init__()
        self.concat = concat
        self.heads = heads
        self.conv = GATv2Conv(
            in_channels=input_dim,
            out_channels=(
                output_dim // heads
                if concat else output_dim
                ),
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat
            )
        self.layer_norm = GraphNorm(output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(
        self, x,
        edge_index,
        edge_attr,
        batch):

        out, (ei, aw) = self.conv(
            x, edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
            )
        weights = aw.mean(dim=-1)
        mask = ei[0] != ei[1]
        self.last_attn_weights = weights[mask]

        out = self.layer_norm(out, batch)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class GINLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        num_lin_layers=2):

        super(GINLayer, self).__init__()

        layers = [nn.Linear(input_dim, output_dim), nn.ReLU()]
        for _ in range(num_lin_layers - 1):
            layers += [nn.Linear(
                output_dim, output_dim), nn.ReLU()]
        self.conv = GINConv(nn.Sequential(*layers))
        self.conv_norm = GraphNorm(output_dim)

    def forward(
        self, x, 
        edge_index, 
        edge_attr, 
        batch):

        x = self.conv(x, edge_index)
        x = self.conv_norm(x, batch)
        return x

    
class MPNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate):

        super(MPNNLayer, self).__init__(aggr='add')
 
        self.message_proj = nn.Sequential(
            nn.Linear(input_dim + edge_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        self.node_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
            )
        self.update_net = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim)
            )
        self.layer_norm = GraphNorm(output_dim)

    def message(self, x_j, edge_attr):
        m = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_proj(m)

    def update(self, aggr_out, x, batch):
        h = self.node_proj(x)
        cat = torch.cat([aggr_out, h], dim=-1)
        new_h = self.update_net(cat)
        out = new_h
        out = self.layer_norm(out, batch)
        return out

    def forward(self, x, edge_index, edge_attr, batch):
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            batch=batch
            )
