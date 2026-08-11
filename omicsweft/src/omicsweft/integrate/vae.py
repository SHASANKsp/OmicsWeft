"""Multi-view Variational Autoencoder (Product-of-Experts) — pure PyTorch.

Each modality has its own encoder producing a Gaussian posterior over a shared
latent space; the joint posterior is their product (PoE), which naturally handles
samples missing a modality (that expert is simply dropped for that sample). Each
modality also has its own decoder. The posterior means form the joint
:class:`Embedding`.

torch is an optional dependency here; the import is guarded so the rest of the
package works without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import INTEGRATORS


def _poe(means: list, logvars: list):
    """Product of Gaussian experts (+ a standard-normal prior expert)."""
    import torch

    # prior expert: mean 0, logvar 0 (precision 1)
    prec = torch.ones_like(means[0])
    mu_prec = torch.zeros_like(means[0])
    for mu, lv in zip(means, logvars):
        p = torch.exp(-lv)
        prec = prec + p
        mu_prec = mu_prec + mu * p
    var = 1.0 / prec
    mu = mu_prec * var
    return mu, torch.log(var + 1e-8)


@INTEGRATORS.register("mvae")
class MultiViewVAE(Integrator):
    def __init__(
        self,
        n_components: int = 16,
        hidden: int = 128,
        epochs: int = 100,
        lr: float = 1e-3,
        beta: float = 1.0,
        samples: str = "common",
        modalities: list[str] | None = None,
        device: str | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
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
            raise ImportError("mvae needs torch: pip install torch") from exc

        torch.manual_seed(self.seed)
        dev = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        mods = self.modalities or data.modalities
        sample_idx = (
            data.sample_names if self.samples == "union"
            else data.common_samples(mods)
        )
        if len(sample_idx) < 4:
            raise ValueError("mvae needs at least 4 samples")

        # build per-modality tensors + presence masks over the sample axis
        Xs, masks, dims = [], [], []
        for m in mods:
            ad = data.get_omic(m)
            df = pd.DataFrame(
                _to_dense(ad.X), index=ad.obs_names, columns=ad.var_names
            ).reindex(sample_idx)
            present = torch.tensor(df.notna().all(axis=1).to_numpy(), device=dev)
            X = torch.tensor(
                np.nan_to_num(df.to_numpy(), nan=0.0), dtype=torch.float32, device=dev
            )
            # standardize present rows
            if present.any():
                mu = X[present].mean(0, keepdim=True)
                sd = X[present].std(0, keepdim=True).clamp_min(1e-6)
                X = (X - mu) / sd
            Xs.append(X)
            masks.append(present)
            dims.append(X.shape[1])

        k, h = self.n_components, self.hidden
        encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2 * k)) for d in dims]
        ).to(dev)
        decoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d)) for d in dims]
        ).to(dev)
        params = list(encoders.parameters()) + list(decoders.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        for epoch in range(self.epochs):
            opt.zero_grad()
            means, logvars = [], []
            for i in range(len(mods)):
                out = encoders[i](Xs[i])
                mu_i, lv_i = out[:, :k], out[:, k:]
                # neutralise absent experts (huge variance -> precision ~0)
                lv_i = torch.where(masks[i][:, None], lv_i, torch.full_like(lv_i, 20.0))
                means.append(mu_i)
                logvars.append(lv_i)
            mu, logvar = _poe(means, logvars)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)

            recon = 0.0
            for i in range(len(mods)):
                xh = decoders[i](z)
                per = ((xh - Xs[i]) ** 2).mean(1)
                recon = recon + (per * masks[i]).sum() / masks[i].sum().clamp_min(1)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + self.beta * kld
            loss.backward()
            opt.step()
            if self.verbose and epoch % 20 == 0:
                print(f"epoch {epoch}: loss={loss.item():.4f}")

        with torch.no_grad():
            means, logvars = [], []
            for i in range(len(mods)):
                out = encoders[i](Xs[i])
                mu_i, lv_i = out[:, :k], out[:, k:]
                lv_i = torch.where(masks[i][:, None], lv_i, torch.full_like(lv_i, 20.0))
                means.append(mu_i)
                logvars.append(lv_i)
            mu, _ = _poe(means, logvars)
            emb = mu.cpu().numpy()

        return Embedding(
            X=emb, samples=pd.Index(sample_idx), method="mvae",
            meta={"modalities": list(mods)},
        )
