import os
import os
import copy
import random
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj
    )
from torch.quasirandom import SobolEngine

from augmentations import apply_rulebook_perturbations
    
from rules import (
    ATOMIC_NUMBER,
    feature_slices
    )
from utils import decode_one_hot
from predictor import predict


def mask_atoms(
    data_raw, idx,
    seed=None,
    all_candidates=False):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    candidates = apply_rulebook_perturbations(
        data_raw,
        idx,
        families=[
            "SP2_POLAR_ALL",
            "SP3_POLAR_ALL",
            "SP2_APOLAR_ALL",
            "SP3_APOLAR_ALL",
            "SP2_REACTIVE_ALL",
            "SP3_REACTIVE_ALL",
            "REDOX_FAMILY",
            "ACYL_FAMILY_ALL",
            "CARBAMATE_FAMILY_ALL",
            "AMIDE_FAMILY_ALL",
            "SULFURE_FAMILY_ALL",
            "PHOSPHORUS_FAMILY_ALL",
            "TOGGLE_CHARGE_FAMILY_ALL",
            "POLYVALENT_FAMILY_ALL",
            "TOGGLE_RING_FAMILY_ALL",
            "RING_FAMILY_ALL",
            "BOND_FAMILY_ALL",
            "DIARYL_FAMILY_ALL"
            ]
        )
    if not candidates:
        if all_candidates:
            return [(copy.deepcopy(data_raw), [idx])]
        return copy.deepcopy(data_raw), [idx]

    if not all_candidates:
        def rule_family(rule_id):
            parts = str(rule_id).split("_")
            if "ALL" in parts:
                idx = parts.index("ALL")
                return "_".join(parts[:idx + 1])
            if "FAMILY" in parts:
                idx = parts.index("FAMILY")
                return "_".join(parts[:idx + 1])
            return parts[0]
        buckets = {}
        for cand in candidates:
            fam = rule_family(cand[2])
            buckets.setdefault(fam, []).append(cand)
        fam_choice = random.choice(list(buckets.keys()))
        data, removed, _rule = random.choice(buckets[fam_choice])
        candidates = [(data, removed, _rule)]
    results = []
    for data, removed, _rule in candidates:
        n0 = data_raw.x.size(0)
        n1 = data.x.size(0)
        group = set()
        changed = []
        changed_adj = []
        if removed:
            group.update(removed)
            if not group:
                group.add(idx)
        else:
            slices = feature_slices()
            a_start, a_end = slices["atomic"]
            min_n = min(n0, n1)
            for j in range(min_n):
                if not torch.equal(
                    data_raw.x[j, a_start:a_end],
                    data.x[j, a_start:a_end]
                ):
                    changed.append(j)
            if n0 == n1:
                def to_adj(edge_index, n):
                    adj = [set() for _ in range(n)]
                    for i, j in edge_index.t().tolist():
                        adj[i].add(j)
                        adj[j].add(i)
                    return adj
                adj_raw = to_adj(data_raw.edge_index, n0)
                adj_new = to_adj(data.edge_index, n1)
                for j in range(min_n):
                    if adj_raw[j] != adj_new[j]:
                        changed.append(j)
                        changed_adj.append(j)
            group.update(changed)
            if not group:
                group.add(idx)
        results.append((data, sorted(group)))
    if all_candidates:
        return results
    return results[0]


def leave1atom(
    model, raw, device,
    num_perturb=10,
    task_idx=0):

    base = copy.deepcopy(raw)
    if hasattr(raw, "y") and raw.y is not None:
        y = raw.y
        if y.dim() == 1:
            y = y.unsqueeze(0)
        base.y = y.to(device)
    else:
        tt = getattr(model, "task_type", None)
        if tt is not None:
            num_tasks = int(tt.numel())
        else:
            num_tasks = getattr(model, "num_tasks", None)
        if num_tasks is None:
            raise ValueError(
                "Cannot infer num_tasks for explanations."
                )
        base.y = torch.zeros(
            (1, num_tasks), dtype=torch.float, device=device
            )
    p0_all, _, _ = predict(
        model,
        DataLoader([base], batch_size=1),
        device,
        return_embeddings=False
        )
    p0 = p0_all[0, task_idx]
    n = raw.x.size(0)
    mean = np.zeros(n)
    std = np.zeros(n)
    pos = np.zeros(n)
    neg = np.zeros(n)
    trim = 0.1
    per_node = [[] for _ in range(n)]

    for i in range(n):
        sobol = SobolEngine(1, scramble=True)
        seq = sobol.draw(num_perturb).squeeze()
        for v in seq:
            seed = int(v.item() * (2**32 - 1))
            all_masks = mask_atoms(raw, i, seed=seed, all_candidates=True)
            for masked, group_nodes in all_masks:
                if (torch.equal(raw.x, masked.x)
                        and torch.equal(raw.edge_index, masked.edge_index)
                        and torch.equal(raw.edge_attr, masked.edge_attr)):
                    continue
                if hasattr(raw, "y") and raw.y is not None:
                    y_local = raw.y
                    if y_local.dim() == 1:
                        y_local = y_local.unsqueeze(0)
                    masked.y = y_local.to(device)
                else:
                    masked.y = torch.zeros(
                        (1, num_tasks), dtype=torch.float,
                        device=device
                        )
                pred2_all, _, _ = predict(
                    model, DataLoader([masked], batch_size=1),
                    device, return_embeddings=False
                    )
                p2 = pred2_all[0, task_idx]
                delta = float(p0 - p2)
                if not group_nodes:
                    group_nodes = [i]
                share = delta / max(len(group_nodes), 1)
                for node_idx in group_nodes:
                    if node_idx < n:
                        per_node[node_idx].append(share)

    for i in range(n):
        arr = np.array(per_node[i], dtype=float)
        if arr.size == 0:
            continue
        k = int(len(arr) * trim)
        if len(arr) > 2 * k:
            arr = np.sort(arr)[k:-k]
        mean[i] = arr.mean()
        std[i] = arr.std(ddof=0)
        pos[i] = np.mean(arr > 0)
        neg[i] = np.mean(arr < 0)

    return mean, std, pos, neg


def plot_contours(
    index, smiles, 
    importance, 
    out_path):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, importance,
        alpha=0.5, 
        contourLines=6,
        colorMap='bwr', 
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(coll.get_facecolor())
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        f"{out_path}/{index}.svg",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_explanations(
    model, 
    loader, 
    device, 
    out_path,
    num_perturb=10,
    task_idx=0):

    model.to(device)
    model.eval()
    index = 1
    with torch.no_grad():
        for batch in loader:
            raw = copy.deepcopy(batch)
            smiles = raw.smiles
            batch = batch.to(device)
            bi = batch.batch
            src, dst = batch.edge_index
            counts = torch.bincount(bi)
            counts = counts.cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                mask = (bi[src] == i) & (bi[dst] == i)
                ei = batch.edge_index[:, mask].cpu().clone()
                ei[0] -= start
                ei[1] -= start
                ea = batch.edge_attr[mask].cpu().clone()

                data = Data(
                    x=raw.x[start:start+n].cpu().clone(),
                    edge_index=ei,
                    edge_attr=ea,
                    smiles=smiles[i]
                    )
                mean, _, pos, neg = leave1atom(
                    model, data, device,
                    num_perturb=num_perturb,
                    task_idx=task_idx
                    )
                weight = pos - neg
                node_imp = mean * weight
                imp = node_imp.copy()
                nz_mask = np.abs(imp) > 1e-12
                if np.any(nz_mask):
                    nz_mean = imp[nz_mask].mean()
                    nz_std = imp[nz_mask].std()
                    if nz_std > 1e-12:
                        imp[nz_mask] = (
                            imp[nz_mask] - nz_mean) / (nz_std + 1e-8)
                    imp[~nz_mask] = 0.0
                plot_contours(
                    index, data.smiles,
                    imp, out_path
                    )
                
                start += n
                index += 1


def cluster_attentions(
    weights, 
    edge_index):

    n = weights.size(0)
    raw = weights.sum(0).cpu().numpy() / n
    centered = raw - raw.mean()
    attn = np.zeros(n)
    edges = edge_index.cpu().numpy().T.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = np.where(centered >= 0)[0]
    neg = np.where(centered < 0)[0]

    for idx in [pos, neg]:
        if idx.size:
            for comp in nx.connected_components(
                G.subgraph(idx)):
                m = list(comp)
                attn[m] = centered[m].mean()

    return attn


def plot_attentions(
    out_path, 
    index, 
    smiles, 
    attn):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, attn,
        alpha=0.5,
        contourLines=6,
        colorMap='bwr',
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(
                coll.get_facecolor()
                )
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        f"{out_path}/{index}.svg",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_attentions(
    model, 
    loader, 
    device, 
    out_path):

    model.to(device)
    model.eval()
    index = 1

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            smiles = batch_data.smiles
            x = batch_data.x
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            batch_idx = batch_data.batch

            for layer in model.agg_layers:
                try:
                    out = layer(
                        x, edge_index, edge_attr, batch_idx
                        )
                except TypeError:
                    try:
                        out = layer(x, edge_index, edge_attr)
                    except TypeError:
                        out = layer(x, edge_index)
                x = out[0] if isinstance(out, tuple) else out
            lap_idx, lap_w = get_laplacian(
                edge_index,
                edge_weight=torch.ones(
                    edge_index.size(1), device=device),
                normalization='sym'
                )
            laplacian_matrix = to_dense_adj(
                lap_idx, edge_attr=lap_w,
                max_num_nodes=x.size(0))[0]
            
            adjacency_matrix = torch.eye(
                x.size(0), device=device
                )
            if hasattr(model, 'atom_attention'):
                att_out, attn = model.atom_attention(
                    x,
                    edge_index,
                    batch_idx,
                    laplacian_matrix,
                    adjacency_matrix
                    )
            else:
                last_layer = model.agg_layers[-1]
                alpha = getattr(last_layer, "last_attn_weights", None)
                if alpha is None:
                    raise AttributeError(
                        "view_attentions requires atom_attention or "
                        "last_attn_weights on the last aggregation layer."
                        )
                N = x.size(0)
                att = torch.zeros(N, N, device=device)
                src, dst = edge_index
                att[src, dst] = alpha
                attn = (att + att.t()) * 0.5

            attn = attn.detach().cpu()
            counts = torch.bincount(batch_idx).cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                sub_attn = attn[start:start + n, start:start + n]
                mask = (
                    (batch_idx[edge_index[0]] == i) &
                    (batch_idx[edge_index[1]] == i)
                    )
                sub_ei = edge_index[:, mask].cpu().clone()
                sub_ei[0] -= start
                sub_ei[1] -= start
                att_vals = (sub_attn.sum(0).cpu().numpy() / max(n, 1))
                nz_mask = np.abs(att_vals) > 1e-12
                if np.any(nz_mask):
                    nz_mean = att_vals[nz_mask].mean()
                    nz_std = att_vals[nz_mask].std()
                    if nz_std > 1e-12:
                        att_vals[nz_mask] = (
                            att_vals[nz_mask] - nz_mean) / (nz_std + 1e-8)
                    att_vals[~nz_mask] = 0.0
                plot_attentions(out_path, index, 
                        smiles[i], att_vals
                        )

                start += n
                index += 1
