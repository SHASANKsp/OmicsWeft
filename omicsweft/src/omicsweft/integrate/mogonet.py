"""MOGONET-style supervised integrator — pure PyTorch (dense GCN, no PyG).

Faithful to the MOGONET design (Wang et al., 2021): build a sample-similarity
graph per omic, run an omics-specific graph convolutional network, and fuse the
per-omic label predictions with a View Correlation Discovery Network (VCDN).
This is a *supervised* integrator: it needs a ``label_key``. The concatenated
pre-fusion GCN representations are returned as the joint :class:`Embedding`, so
any downstream task can consume it.

Note on leakage: because this uses labels, evaluating it by predicting the *same*
labels is circular. Use it to embed for a *different* target, or hold labels out.
The benchmark layer flags supervised integrators accordingly.

Implemented with dense adjacency matrices (fine for cohort-scale sample graphs);
no torch-geometric dependency. torch is optional / guarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import INTEGRATORS

supervised = True  # marker read by the benchmark layer


def _cosine_adj(X, k: int):
    import torch

    Xn = torch.nn.functional.normalize(X, dim=1)
    sim = Xn @ Xn.T
    # keep top-k neighbours per node, symmetrize, add self-loops
    n = sim.shape[0]
    k = min(k, n - 1)
    thresh = sim.topk(k + 1, dim=1).values[:, -1:]
    adj = (sim >= thresh).float()
    adj = ((adj + adj.T) > 0).float()
    adj.fill_diagonal_(1.0)
    # symmetric normalization D^-1/2 A D^-1/2
    deg = adj.sum(1)
    dinv = torch.diag(deg.clamp_min(1e-8).pow(-0.5))
    return dinv @ adj @ dinv


@INTEGRATORS.register("mogonet")
class MOGONET(Integrator):
    def __init__(
        self,
        label_key: str,
        hidden: int = 64,
        embed_dim: int = 32,
        k_neighbors: int = 10,
        epochs: int = 200,
        lr: float = 5e-3,
        samples: str = "common",
        modalities: list[str] | None = None,
        device: str | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.label_key = label_key
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        self.epochs = epochs
        self.lr = lr
        self.samples = samples
        self.modalities = modalities
        self.device = device
        self.seed = seed
        self.verbose = verbose

    def fit_transform(self, data: OmicsData) -> Embedding:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("mogonet needs torch: pip install torch") from exc

        torch.manual_seed(self.seed)
        dev = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        mods = self.modalities or data.modalities
        samples = data.common_samples(mods)

        y_raw = data.labels(self.label_key).reindex(samples)
        mask = y_raw.notna().to_numpy()
        if mask.sum() < 4:
            raise ValueError("mogonet needs >=4 labelled shared samples")
        y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
        n_classes = len(np.unique(y))
        y_t = torch.tensor(y, dtype=torch.long, device=dev)
        train_idx = torch.tensor(np.where(mask)[0], device=dev)

        # per-omic features, standardized, + normalized adjacency
        Xs, adjs, dims = [], [], []
        for m in mods:
            X = _to_dense(data.get_omic(m)[samples].X)
            X = np.nan_to_num(X, nan=0.0)
            mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
            X = (X - mu) / np.clip(sd, 1e-6, None)
            Xt = torch.tensor(X, dtype=torch.float32, device=dev)
            Xs.append(Xt)
            adjs.append(_cosine_adj(Xt, self.k_neighbors))
            dims.append(Xt.shape[1])

        class GCN(nn.Module):
            def __init__(self, d_in, h, d_out):
                super().__init__()
                self.w1 = nn.Linear(d_in, h)
                self.w2 = nn.Linear(h, d_out)
                self.act = nn.ReLU()

            def forward(self, x, a):
                x = self.act(a @ self.w1(x))
                return self.act(a @ self.w2(x))

        gcns = nn.ModuleList(
            [GCN(d, self.hidden, self.embed_dim) for d in dims]
        ).to(dev)
        cls_heads = nn.ModuleList(
            [nn.Linear(self.embed_dim, n_classes) for _ in dims]
        ).to(dev)
        # VCDN: fuse per-omic class-probability outer product tensor
        vcdn = nn.Sequential(
            nn.Linear(n_classes ** len(mods), self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, n_classes),
        ).to(dev)

        params = (
            list(gcns.parameters())
            + list(cls_heads.parameters())
            + list(vcdn.parameters())
        )
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=5e-4)
        ce = nn.CrossEntropyLoss()

        def forward():
            embs = [gcns[i](Xs[i], adjs[i]) for i in range(len(mods))]
            logits = [cls_heads[i](embs[i]) for i in range(len(mods))]
            probs = [torch.softmax(lg, dim=1) for lg in logits]
            # outer product across views -> VCDN input
            fused = probs[0]
            for p in probs[1:]:
                fused = torch.einsum("n...,nc->n...c", fused, p)
            fused = fused.reshape(fused.shape[0], -1)
            return embs, logits, vcdn(fused)

        for epoch in range(self.epochs):
            opt.zero_grad()
            embs, logits, vlogits = forward()
            loss = ce(vlogits[train_idx], y_t)
            for lg in logits:
                loss = loss + ce(lg[train_idx], y_t)
            loss.backward()
            opt.step()
            if self.verbose and epoch % 40 == 0:
                print(f"epoch {epoch}: loss={loss.item():.4f}")

        with torch.no_grad():
            embs, _, _ = forward()
            joint = torch.cat(embs, dim=1).cpu().numpy()

        return Embedding(
            X=joint, samples=pd.Index(samples), method="mogonet",
            meta={"modalities": list(mods), "supervised": True,
                  "label_key": self.label_key},
        )
