"""Hypersphere primitives: projection, vMF / truncated vMF sampling, Bessel ratios, hypercones.

All tensors are batched along dim 0 and live on S^{D-1} along the last (flattened) dim.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def flat(x):
    return x.reshape(x.shape[0], -1)


def project(x, eps=1e-12):
    """Pi(x) = x / ||x||, applied per sample over all non-batch dims."""
    n = flat(x).norm(dim=1).clamp_min(eps)
    return x / n.view(-1, *([1] * (x.dim() - 1)))


def bdot(a, b):
    return (flat(a) * flat(b)).sum(dim=1)


def _bview(v, x):
    return v.view(-1, *([1] * (x.dim() - 1)))


def tangent(z, v):
    """Project v onto the tangent space of the sphere at z: v - (v.z) z."""
    return v - _bview(bdot(v, z), z) * z


def angle(a, b):
    return bdot(project(a), project(b)).clamp(-1.0, 1.0).acos()


def rotate_towards(p, u, theta):
    """Point at geodesic angle `theta` from unit `p` along unit tangent direction `u`."""
    theta = _bview(torch.as_tensor(theta, dtype=p.dtype, device=p.device).expand(p.shape[0]), p)
    return project(theta.cos() * p + theta.sin() * u)


def random_tangent(p):
    return project(tangent(p, torch.randn_like(p)))


def sample_vmf(mu, kappa, max_iter=1000):
    """Draw one sample per row from vMF(mu, kappa) on S^{D-1} (Wood, 1994).

    mu: (B, ...) unit vectors. kappa: scalar or (B,) tensor, kappa >= 0 (0 = uniform).
    """
    B = mu.shape[0]
    m = flat(mu).shape[1]
    dev = mu.device
    k = torch.as_tensor(kappa, dtype=torch.float64, device=dev).expand(B).clone()
    # Numerically stable form of b = (-2k + sqrt(4k^2 + (m-1)^2)) / (m-1).
    b = (m - 1) / (2 * k + torch.sqrt(4 * k * k + (m - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = k * x0 + (m - 1) * torch.log1p(-x0 * x0)

    w = torch.zeros(B, dtype=torch.float64, device=dev)
    todo = torch.ones(B, dtype=torch.bool, device=dev)
    beta = torch.distributions.Beta((m - 1) / 2.0, (m - 1) / 2.0)
    for _ in range(max_iter):
        idx = todo.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        zb = beta.sample((idx.numel(),)).to(dev, torch.float64)
        bi, x0i, ci, ki = b[idx], x0[idx], c[idx], k[idx]
        wi = (1 - (1 + bi) * zb) / (1 - (1 - bi) * zb)
        u = torch.rand(idx.numel(), dtype=torch.float64, device=dev)
        ok = ki * wi + (m - 1) * torch.log1p(-x0i * wi) - ci >= torch.log(u)
        w[idx[ok]] = wi[ok]
        todo[idx[ok]] = False
    if todo.any():
        raise RuntimeError("vMF rejection sampler did not converge")

    w = w.to(mu.dtype).clamp(-1.0, 1.0)
    v = random_tangent(mu)
    return project(_bview(w, mu) * mu + _bview((1 - w * w).clamp_min(0).sqrt(), mu) * v)


def sample_tvmf(mu, kappa, theta_min=0.0, theta_max=math.pi, grid=4096, chunk=1024):
    """Truncated vMF: vMF(mu, kappa) restricted to the angular band theta_min <= angle(z, mu) <= theta_max.

    [0, theta] is the class hypercone of Sec. 4.3 (TvMF with the cone centred on the mean);
    [theta, theta + delta] is the shell just outside it. Exact up to the grid: the vMF factorises
    into the angle phi to mu, with density ~ exp(kappa cos phi) sin^{D-2} phi, and a uniform
    tangent direction, so phi is drawn by inverse CDF on [theta_min, theta_max].
    mu: (B, ...) unit vectors; kappa, theta_min, theta_max: scalars or (B,) tensors.
    """
    B = mu.shape[0]
    m = flat(mu).shape[1]
    dev = mu.device
    par = [torch.as_tensor(p, dtype=torch.float64, device=dev).expand(B) for p in (kappa, theta_min, theta_max)]
    k, lo, hi = par
    if (lo < -1e-6).any() or (hi > math.pi + 1e-6).any() or (lo >= hi).any():
        raise ValueError("need 0 <= theta_min < theta_max <= pi")
    lo, hi = lo.clamp(min=0.0), hi.clamp(max=math.pi)            # float32 rounding of the bounds
    s = torch.linspace(0, 1, grid + 1, dtype=torch.float64, device=dev)
    phi = torch.empty(B, dtype=torch.float64, device=dev)
    for i in range(0, B, chunk):
        ki, li, hi_ = k[i:i + chunk, None], lo[i:i + chunk, None], hi[i:i + chunk, None]
        g = li + (hi_ - li) * s                                    # (b, grid+1)
        logp = ki * g.cos() + (m - 2) * g.sin().clamp_min(1e-300).log()
        p = (logp - logp.max(dim=1, keepdim=True).values).exp()
        cdf = torch.cat([torch.zeros_like(p[:, :1]), ((p[:, 1:] + p[:, :-1]) / 2).cumsum(1)], 1)
        cdf = cdf / cdf[:, -1:]
        u = torch.rand(g.shape[0], 1, dtype=torch.float64, device=dev)
        j = torch.searchsorted(cdf, u).clamp(1, grid)
        c0, c1 = cdf.gather(1, j - 1), cdf.gather(1, j)
        g0, g1 = g.gather(1, j - 1), g.gather(1, j)
        phi[i:i + chunk] = (g0 + (g1 - g0) * ((u - c0) / (c1 - c0).clamp_min(1e-300))).squeeze(1)
    phi = torch.minimum(torch.maximum(phi, lo), hi).to(mu.dtype)
    return rotate_towards(mu, random_tangent(mu), phi)


def bessel_ratio(kappa, dim):
    """A_D(kappa) = I_{D/2}(kappa) / I_{D/2-1}(kappa) = E[mu^T z] for z ~ vMF(mu, kappa) on S^{D-1}.

    Backward continued fraction R_nu = kappa / (2(nu+1) + kappa R_{nu+1}), seeded with the
    Amos-type approximation so it is accurate for any kappa / order. Works on numpy arrays.
    """
    k = np.asarray(kappa, dtype=np.float64)
    nu = dim / 2.0 - 1.0
    depth = 200
    top = nu + depth
    r = k / (top + 1 + np.sqrt(k * k + (top + 1) ** 2))
    for j in range(depth, 0, -1):
        r = k / (2 * (nu + j) + k * r)
    return r


def inverse_bessel_ratio(rho, dim, iters=100):
    """Solve A_D(kappa) = rho for kappa >= 0 by bisection in log-space."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1 - 1e-12)
    lo = np.full_like(rho, -20.0)
    hi = np.full_like(rho, math.log(1e12))
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_small = bessel_ratio(np.exp(mid), dim) < rho
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return np.where(rho <= 0, 0.0, np.exp((lo + hi) / 2))


@dataclass
class Hypercones:
    """Class hypercones: unit class means mu_k and half-angles theta_k = `quantile` of the angles to mu_k."""
    mu: torch.Tensor     # (K, d)
    theta: torch.Tensor  # (K,)

    @classmethod
    def fit(cls, feats, labels, num_classes, quantile=0.95):
        feats = F.normalize(feats.float(), dim=1)
        mu = torch.stack([F.normalize(feats[labels == k].mean(0), dim=0) for k in range(num_classes)])
        ang = (feats * mu[labels]).sum(1).clamp(-1, 1).acos()
        theta = torch.stack([ang[labels == k].quantile(quantile) for k in range(num_classes)])
        return cls(mu, theta)
