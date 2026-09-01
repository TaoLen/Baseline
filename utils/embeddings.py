import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from params import load_embeddings
from figures import (
    FigureConfig,
    SaveConfig,
    RcParamsConfig,
    Palette,
    style_spines,
    legend_frame
    )


def normalize_embeddings(embeddings, jitter_std=1e-4):
    X = np.asarray(embeddings, dtype=float)
    if jitter_std > 0:
        noise = np.random.normal(
            loc=0.0, 
            scale=jitter_std, 
            size=X.shape
        )
        X = X + noise
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n



def embeddings2tSNE(
    embeddings, 
    perplexity,
    learning_rate, 
    seed):

    tsne = TSNE(n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=1000,
        random_state=seed,
        init='pca'
        )
    dimension = tsne.fit_transform(embeddings)

    return dimension


def embeddings2uMAP(
    embeddings,
    n_neighbors, 
    min_dist, 
    seed):

    umap = UMAP(n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed
        )
    dimension = umap.fit_transform(embeddings)

    return dimension


def plot_embeddings(
    reduced,
    labels,
    file_path,
    title,
    x_label,
    y_label,
    cls_type=None,
    palette=Palette(),
    rc=RcParamsConfig(),
    save_cfg=None,
    bins=30,
    fig_scale=None,
    dpi=600):

    rc.apply()
    if save_cfg is None:
        save_cfg = SaveConfig(
            out_path=file_path,
            dpi=dpi, suffix="_embed")

    fig_cfg = FigureConfig(fig_scale=fig_scale)
    side = fig_cfg.side()
    fig = plt.figure(figsize=(side, side))
    gs = plt.GridSpec(nrows=2,
        ncols=1,height_ratios=[4, 1.0], hspace=0.06
        )
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax)
    lab = (labels.numpy() if torch.is_tensor(labels)
           else labels)
    lab = np.asarray(lab).squeeze()
    if lab.ndim > 1:
        lab = lab.reshape(-1)
    lab_float = lab.astype(float, copy=False)
    valid = ~np.isnan(lab_float)
    x = reduced[:, 0]
    y = reduced[:, 1]
    vmin, vmax = float(np.min(x)), float(np.max(x))
    span = max(vmax - vmin, 1e-9)
    if np.isfinite(vmin) and np.isfinite(vmax):
        lo = vmin - 0.08 * span
        hi = vmax + 0.08 * span
    else:
        lo, hi = 0.0, 1.0
    if cls_type is True:
        if lab.dtype.kind in ("f", "c"):
            bin_lab = (lab_float >= 0.5).astype(int)
        else:
            bin_lab = lab.astype(int, copy=False)

        ax.scatter(x[~valid], y[~valid],
            c="lightgray", s=6, alpha=0.4
            )
        bin_valid = bin_lab[valid]
        colors = np.where(
            bin_valid == 0,
            palette.inact,
            palette.act
            )
        ax.scatter(x[valid], y[valid], c=colors,
            s=9, alpha=0.8
            )
        hdl = [plt.Line2D(
                [0], [0], marker="o",
                linestyle="", label="Inactive",
                markerfacecolor=palette.inact,
                markeredgecolor="none",
                markersize=6, alpha=0.8),
            plt.Line2D([0], [0], marker="o",
                linestyle="", label="Active",
                markerfacecolor=palette.act,
                markeredgecolor="none",
                markersize=6, alpha=0.8),
                ]
        leg = ax.legend(handles=hdl,
            loc="upper left", frameon=True,
            fontsize=rc.legend_fontsize
            )
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_edgecolor("black")
        fr.set_linewidth(1.0)

        m0, m1 = (bin_valid == 0), (bin_valid == 1)
        ax_hist.hist(x[valid][m0],
            bins=bins, range=(lo, hi),
            color=palette.inact, alpha=0.60,
            label="Inactive"
            )
        ax_hist.hist(x[valid][m1],
            bins=bins, range=(lo, hi),
            color=palette.act, alpha=0.65,
            label="Active"
            )
        legend_frame(ax_hist, fontsize=rc.legend_fontsize)

    elif cls_type is False:
        ax.scatter(x[~valid], y[~valid],
            c="lightgray", s=6, alpha=0.4
            )
        sc = ax.scatter(x[valid], y[valid],
            c=lab_float[valid], s=9, alpha=0.8,
            cmap=palette.reg_cmap
            )
        cb = fig.colorbar(sc, ax=[ax, ax_hist],
            location="right", fraction=0.05,
            pad=0.03, use_gridspec=True
            )
        cb.ax.yaxis.set_label_position("right")
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.tick_params(labelsize=rc.legend_fontsize)
        cb.set_label("Label")

        vals = lab_float[valid]
        qs = np.nanpercentile(vals, [0, 33.333, 66.666, 100])
        m_low = (vals >= qs[0]) & (vals <= qs[1])
        m_mid = (vals > qs[1]) & (vals <= qs[2])
        m_high = (vals > qs[2]) & (vals <= qs[3])

        ax_hist.hist(x[valid][m_low], bins=bins,
            range=(lo, hi), color=palette.inact,
            alpha=0.65, label="Low"
            )
        ax_hist.hist(x[valid][m_mid], bins=bins,
            range=(lo, hi), color=palette.mid,
            alpha=0.75, label="Mid"
            )
        ax_hist.hist(x[valid][m_high], bins=bins,
            range=(lo, hi), color=palette.act,
            alpha=0.65, label="High"
            )
        legend_frame(ax_hist, fontsize=rc.legend_fontsize)
    else:
        ax.scatter(x, y, c="gray", s=9, alpha=0.8)
    ax.set_xlim(lo, hi)
    ax_hist.set_xlim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax_hist.set_xlabel(x_label)
    ax_hist.set_ylabel("Count")
    style_spines(ax)
    style_spines(ax_hist)
    ax.grid(False)
    ax_hist.grid(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.tight_layout()
    save_cfg.maybe_save(fig)
    plt.show()


def visualize_embeddings(
    in_path,
    epoch,
    method,
    task_index=None,
    out_path=None,
    perplexity=30,
    learning_rate=200,
    n_neighbors=15,
    min_dist=0.1,
    seed=42,
    fig_scale=None,
    dpi=600,
    palette=Palette(),
    rc=RcParamsConfig()):

    loaded = load_embeddings(in_path, epoch)
    if len(loaded) == 3:
        embeddings, labels, is_cls = loaded
    else:
        embeddings, labels = loaded
        is_cls = None
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if torch.is_tensor(labels) and labels.dim() > 1:
        if task_index is None:
            raise ValueError("Please specify task_index")
        labels = labels[:, task_index]
    labels_np = (labels.tolist()
                 if torch.is_tensor(labels)
                 else labels)
    emb = embeddings.numpy()
    emb = normalize_embeddings(emb)

    if method == "tSNE":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=1000,
            random_state=seed,
            init="pca")
        reduced = reducer.fit_transform(emb)
    elif method == "uMAP":
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed)
        reduced = reducer.fit_transform(emb)
    else:
        raise ValueError(
            "Method must be 'tSNE' or 'uMAP'")
    cls_type = None
    if is_cls is not None and task_index is not None:
        cls_type = (
            bool(is_cls[task_index].item())
            if torch.is_tensor(is_cls)
            else bool(is_cls[task_index]))
    plot_embeddings(
        reduced=reduced,
        labels=labels_np,
        file_path=out_path,
        title=(f"{method} Embedding "
               f"(Epoch {epoch+1})"),
        x_label=(f"{method} Dim 1"),
        y_label=(f"{method} Dim 2"),
        cls_type=cls_type, palette=palette, rc=rc,
        save_cfg=SaveConfig(
            out_path=out_path,
            dpi=dpi,
            suffix="_embed",),
        fig_scale=fig_scale,
        dpi=dpi)