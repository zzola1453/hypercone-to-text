import math

import numpy as np
import pytest
import torch

from sphere import Hypercones, angle, bdot, bessel_ratio, inverse_bessel_ratio, project, sample_tvmf, sample_vmf


def test_bessel_ratio_matches_closed_form_d3():
    # On S^2: A_3(k) = coth(k) - 1/k.
    k = np.array([0.1, 1.0, 5.0, 50.0, 500.0])
    np.testing.assert_allclose(bessel_ratio(k, 3), 1 / np.tanh(k) - 1 / k, rtol=1e-6)


@pytest.mark.parametrize("dim", [3, 16, 128, 768])
def test_inverse_bessel_ratio_roundtrip(dim):
    rho = np.array([0.01, 0.3, 0.7, 0.99])
    np.testing.assert_allclose(bessel_ratio(inverse_bessel_ratio(rho, dim), dim), rho, rtol=1e-6)


@pytest.mark.parametrize("dim,kappa", [(3, 0.0), (3, 10.0), (16, 5.0), (128, 60.0), (768, 300.0)])
def test_vmf_mean_cosine(dim, kappa):
    torch.manual_seed(0)
    n = 4000
    mu = project(torch.randn(n, dim))
    z = sample_vmf(mu, kappa)
    assert torch.allclose(z.norm(dim=1), torch.ones(n), atol=1e-5)
    assert bdot(z, mu).mean().item() == pytest.approx(float(bessel_ratio(kappa, dim)), abs=0.02)


@pytest.mark.parametrize("dim,kappa", [(3, 10.0), (16, 5.0), (768, 300.0)])
def test_tvmf_untruncated_matches_vmf(dim, kappa):
    torch.manual_seed(0)
    mu = project(torch.randn(4000, dim))
    z = sample_tvmf(mu, kappa)
    assert torch.allclose(z.norm(dim=1), torch.ones(4000), atol=1e-5)
    assert bdot(z, mu).mean().item() == pytest.approx(float(bessel_ratio(kappa, dim)), abs=0.02)


@pytest.mark.parametrize("lo,hi", [(0.0, 0.9), (0.9, 1.3)])
def test_tvmf_matches_rejected_vmf(lo, hi):
    # Low D, so plain rejection from vMF is feasible as a reference.
    torch.manual_seed(0)
    dim, kappa = 16, 5.0
    mu = project(torch.randn(1, dim)).expand(60000, dim)
    ref = angle(sample_vmf(mu, kappa), mu)
    ref = ref[(ref >= lo) & (ref <= hi)]
    a = angle(sample_tvmf(mu[:len(ref)], kappa, lo, hi), mu[:len(ref)])
    assert a.min() >= lo - 1e-4 and a.max() <= hi + 1e-4
    q = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(a.quantile(q), ref.quantile(q), atol=0.01)


def test_tvmf_shell_outside_cone_high_dim():
    # Virtual outliers: kappa from the cone radius, truncated to the shell just outside the cone.
    torch.manual_seed(0)
    dim, theta, delta = 768, 1.0, 0.1
    kappa = float(inverse_bessel_ratio(math.cos(theta), dim))
    mu = project(torch.randn(2000, dim))
    a = angle(sample_tvmf(mu, kappa, theta, theta + delta), mu)
    assert a.min() >= theta - 1e-4 and a.max() <= theta + delta + 1e-4


def test_hypercones_fit_covers_quantile():
    torch.manual_seed(0)
    mu = project(torch.randn(3, 16))
    labels = torch.arange(3).repeat_interleave(500)
    feats = sample_vmf(mu[labels], 50.0)
    cones = Hypercones.fit(feats, labels, 3, quantile=0.95)
    inside = angle(feats, cones.mu[labels]) <= cones.theta[labels]
    assert inside.float().mean().item() == pytest.approx(0.95, abs=0.01)
