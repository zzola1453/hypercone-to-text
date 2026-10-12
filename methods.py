"""Open-set classifiers on frozen sentence embeddings. All share the same MLP body.

Every method returns (pred, score) on the test set: pred in {0..K-1} or K (= unknown / rejected),
score higher = more likely known (for AUROC / FPR95).

- msp:        softmax classifier, reject if max prob < 5% quantile of validation max probs.
- adb:        ADB (Zhang et al., 2021): Euclidean balls around CE-feature centroids, radii learned with
              the ADB loss. Re-implemented on our embeddings, not the official BERT pipeline.
- cone_q:     hypercone classifier without outliers: cosine-softmax features, cone radius = 95% quantile
              of the training angles (the HCR cones of HyperSphereDiff).
- cone_<syn>: hypercones with learned radii theta_k, trained with virtual outliers <syn>:
    tvmf  (ours) vMF truncated to the shell [r_k, r_k + delta] just outside the cone, kappa from A_d(kappa) = cos r_k
    vmf          untruncated vMF with kappa from A_d(kappa) = cos(r_k + delta / 2)
    gauss        VOS-style: lowest-likelihood draws of a class-conditional Gaussian (shared covariance)
  r_k is the q-quantile of the training angles to the class prototype, refreshed every epoch.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sphere import inverse_bessel_ratio, project, sample_tvmf, sample_vmf

EPS = 1e-6


def body(din, d, hidden=512):
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, d))


def batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    return [perm[i:i + bs] for i in range(0, n, bs)]


# ---------------------------------------------------------------- softmax / MSP / ADB

class SoftmaxNet(nn.Module):
    def __init__(self, din, K, d=128):
        super().__init__()
        self.body, self.fc = body(din, d), nn.Linear(d, K)

    def forward(self, x):
        return self.fc(self.body(x))


def train_softmax(x, y, K, epochs, seed, bs=128, lr=1e-3):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = SoftmaxNet(x.shape[1], K).to(x.device)
    opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(epochs):
        net.train()
        for b in batches(len(x), bs, gen):
            loss = F.cross_entropy(net(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def msp(net, xval, xte, K, tpr=0.95):
    tau = F.softmax(net(xval), 1).max(1).values.quantile(1 - tpr)
    p, pred = F.softmax(net(xte), 1).max(1)
    return torch.where(p >= tau, pred, torch.full_like(pred, K)), p


def _accept(score_val, score_te, pred, K, tpr=0.95):
    """Reject when the score is below the level that accepts `tpr` of the (known-only) validation set."""
    level = score_val.quantile(1 - tpr)
    return torch.where(score_te >= level, pred, torch.full_like(pred, K)), score_te


@torch.no_grad()
def energy(net, xval, xte, K, tpr=0.95):
    """Energy score (Liu et al., NeurIPS 2020): logsumexp of the logits, higher = known."""
    lv, lt = net(xval), net(xte)
    return _accept(torch.logsumexp(lv, 1), torch.logsumexp(lt, 1), lt.argmax(1), K, tpr)


@torch.no_grad()
def mahalanobis(net, x, y, xval, xte, K, tpr=0.95, feat=None):
    """Mahalanobis distance (Lee et al., NeurIPS 2018) on the penultimate features: class means with a
    shared covariance; predict the nearest class, score = minus the smallest distance.
    `feat` replaces the feature map (default: the network body)."""
    feat = feat or net.body
    f = feat(x)
    mu = torch.stack([f[y == k].mean(0) for k in range(K)])
    prec = torch.linalg.pinv(torch.cov((f - mu[y]).T))

    def dist(z):
        d = z[:, None, :] - mu[None]                          # (N, K, d)
        return torch.einsum("nkd,de,nke->nk", d, prec, d)

    dv, dt = dist(feat(xval)), dist(feat(xte))
    return _accept(-dv.min(1).values, -dt.min(1).values, dt.argmin(1), K, tpr)


@torch.no_grad()
def knn(net, x, xval, xte, K, k=10, tpr=0.95, feat=None, logits=None):
    """Deep nearest neighbours (Sun et al., ICML 2022): L2-normalised penultimate features, score = minus the
    distance to the k-th nearest training feature; the class comes from the classifier head.
    `feat` / `logits` replace the feature map and the head (default: the softmax network)."""
    feat, logits = feat or net.body, logits or net
    bank = F.normalize(feat(x), dim=1)

    def score(z):
        return -torch.cdist(F.normalize(feat(z), dim=1), bank).topk(k, largest=False).values[:, -1]

    return _accept(score(xval), score(xte), logits(xte).argmax(1), K, tpr)


@torch.no_grad()
def cosine_msp(net, xval, xte, K, scale=16.0, tpr=0.95):
    """MSP of a ConeNet's cosine-softmax head, softmax(scale * cos(z, mu_k))."""
    p = lambda x: F.softmax(scale * net.feats(x) @ net.protos().T, 1).max(1)
    (pv, _), (pt, pred) = p(xval), p(xte)
    return _accept(pv, pt, pred, K, tpr)


def adb(net, x, y, xte, K, epochs=100, bs=128, lr=0.05, seed=0, xval=None, tpr=0.95):
    """With `xval`, the accept level is calibrated on known validation data as in `cone_predict`
    (the ADB loss alone drives each radius to the median training distance)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        f, fte = net.body(x), net.body(xte)
        c = torch.stack([f[y == k].mean(0) for k in range(K)])
    raw = nn.Parameter(torch.randn(K, device=x.device))
    opt = torch.optim.Adam([raw], lr)
    for _ in range(epochs):
        for b in batches(len(x), bs, gen):
            d = (f[b] - c[y[b]]).norm(dim=1)
            r = F.softplus(raw)[y[b]]
            out = (d > r).float()
            loss = (out * (d - r) + (1 - out) * (r - d)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        m = F.softplus(raw) - torch.cdist(fte, c)            # (N, K) > 0 inside the ball
        s, pred = m.max(1)
        level = 0.0
        if xval is not None:
            level = (F.softplus(raw) - torch.cdist(net.body(xval), c)).max(1).values.quantile(1 - tpr)
    return torch.where(s >= level, pred, torch.full_like(pred, K)), s


# ---------------------------------------------------------------- hypercones

class ConeNet(nn.Module):
    def __init__(self, din, K, d=128, theta0=1.0):
        super().__init__()
        self.body = body(din, d)
        self.proto = nn.Parameter(torch.randn(K, d))
        self.raw_theta = nn.Parameter(torch.full((K,), math.log(theta0 / (math.pi - theta0))))

    def feats(self, x):
        return F.normalize(self.body(x), dim=1)

    def protos(self):
        return F.normalize(self.proto, dim=1)

    def theta(self):
        return math.pi * torch.sigmoid(self.raw_theta)

    def angles(self, z):
        return (z @ self.protos().T).clamp(-1 + EPS, 1 - EPS).acos()   # (N, K)


@torch.no_grad()
def class_radius(net, x, y, K, q):
    """q-quantile of the training angles of each class to its prototype (the HCR cone radius)."""
    a = net.angles(net.feats(x)).gather(1, y[:, None]).squeeze(1)
    return torch.stack([a[y == k].quantile(q) for k in range(K)])


class OutlierSynth:
    """Virtual outliers around the class cones of unit vectors z (features or input embeddings)."""

    def __init__(self, kind, delta):
        self.kind, self.delta = kind, delta

    @torch.no_grad()
    def refresh(self, z, y, mu, K, q):
        """z: (N, d) unit vectors, mu: (K, d) unit cone axes. Cone radius r_k = q-quantile of angles."""
        self.mu, d = mu, z.shape[1]
        a = (z * mu[y]).sum(1).clamp(-1 + EPS, 1 - EPS).acos()
        self.r = torch.stack([a[y == k].quantile(q) for k in range(K)]).clamp(max=math.pi - 1e-3)
        r = self.r.cpu().numpy()
        if self.kind == "tvmf":
            self.kappa = torch.as_tensor(inverse_bessel_ratio(np.cos(r), d), dtype=torch.float32, device=z.device)
        elif self.kind == "vmf":
            mid = np.minimum(r + self.delta / 2, math.pi / 2)
            self.kappa = torch.as_tensor(inverse_bessel_ratio(np.cos(mid), d), dtype=torch.float32, device=z.device)
        elif self.kind == "gauss":
            means = torch.stack([z[y == k].mean(0) for k in range(K)])
            cov = torch.cov((z - means[y]).T) + 1e-4 * torch.eye(d, device=z.device)
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(means[0]), covariance_matrix=cov)
            bank = []
            for k in range(K):                                  # VOS: keep the lowest-likelihood 1% of draws
                s = dist.sample((2000,))
                keep = dist.log_prob(s).argsort()[:20]
                bank.append(project(means[k] + s[keep]))
            self.bank = torch.stack(bank)                       # (K, 20, d)

    @torch.no_grad()
    def sample(self, ks):
        if self.kind == "tvmf":
            hi = (self.r[ks] + self.delta).clamp(max=math.pi)
            return sample_tvmf(self.mu[ks], self.kappa[ks], self.r[ks], hi)
        if self.kind == "vmf":
            return sample_vmf(self.mu[ks], self.kappa[ks])
        j = torch.randint(self.bank.shape[1], (len(ks),), device=ks.device)
        return self.bank[ks, j]


def train_cone(x, y, K, epochs, seed, outliers="tvmf", space="feature", bs=128, lr=1e-3, scale=16.0,
               alpha=1.0, beta=1.0, gamma=0.0, delta=0.1, q=0.95, d=128):
    """Cosine-softmax CE + (with outliers) hinge losses on learned cone radii.

    l_in  = relu(angle(z, mu_y) - theta_y): real samples inside their cone (pulls features in, theta up)
    l_out = relu(max_k theta_k - angle(o, mu_k)): outliers outside every cone
    l_oe  = cross-entropy of the outliers' cosine softmax to the uniform distribution (Outlier Exposure)

    space="feature": outliers are drawn around the cones of the network's own features, once per epoch;
        they are not passed through the network, so they only shape theta and the prototypes.
    space="input": outliers are drawn around the cones of the (unit-norm) input embeddings, fixed, and
        passed through the network, so l_out / l_oe also train the feature map.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = ConeNet(x.shape[1], K, d).to(x.device)
    opt = torch.optim.Adam(net.parameters(), lr)
    syn = OutlierSynth(outliers, delta) if outliers != "none" else None
    if syn is not None and space == "input":
        zin = F.normalize(x, dim=1)
        syn.refresh(zin, y, F.normalize(torch.stack([zin[y == k].mean(0) for k in range(K)]), dim=1), K, q)
    for _ in range(epochs):
        if syn is not None and space == "feature":
            net.eval()
            with torch.no_grad():
                syn.refresh(net.feats(x), y, net.protos(), K, q)
        net.train()
        for b in batches(len(x), bs, gen):
            z = net.feats(x[b])
            cos = z @ net.protos().T
            loss = F.cross_entropy(scale * cos, y[b])
            if syn is not None:
                th = net.theta()
                ang_y = cos.gather(1, y[b][:, None]).squeeze(1).clamp(-1 + EPS, 1 - EPS).acos()
                o = syn.sample(y[b])
                if space == "input":
                    o = net.feats(o)
                ang_o = net.angles(o)
                loss = loss + alpha * F.relu(ang_y - th[y[b]]).mean() \
                    + beta * F.relu((th - ang_o).max(1).values).mean()
                if gamma > 0:
                    loss = loss - gamma * F.log_softmax(scale * (o @ net.protos().T), 1).mean(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    theta = class_radius(net, x, y, K, q) if syn is None else net.theta().detach()
    return net, theta


@torch.no_grad()
def cone_predict(net, theta, xte, K, xval=None, tpr=0.95):
    """Accept if inside some cone (score >= 0). With `xval` (known-only validation data), the accept
    level is instead one global offset so that `tpr` of the validation samples are accepted, as for MSP."""
    m = theta - net.angles(net.feats(xte))                    # (N, K) > 0 inside cone k
    s, pred = m.max(1)
    level = 0.0 if xval is None else (theta - net.angles(net.feats(xval))).max(1).values.quantile(1 - tpr)
    return torch.where(s >= level, pred, torch.full_like(pred, K)), s
