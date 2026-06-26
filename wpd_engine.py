"""
wpd_engine.py — Wasserstein Proximal Descent Engine
====================================================

Theory
------
**Wasserstein-2 Distance (Villani 2009)**

The Wasserstein-2 distance between probability measures mu and nu on R is:

    W_2(mu, nu)^2 = inf_{gamma in Pi(mu,nu)} integral |x-y|^2 d_gamma(x,y)

Where Pi(mu, nu) is the set of couplings (joint distributions with marginals mu and nu).
For empirical distributions on R, this equals the L2 distance between sorted samples.

**Gradient Flows in Wasserstein Space (Jordan, Kinderlehrer & Otto 1998)**

The Fokker-Planck equation:

    d_rho/dt = div(rho * grad(log(rho/rho_target)))

describes the evolution of a distribution rho_t as a gradient flow of the
KL divergence functional F(rho) = KL(rho || rho_target) in the Wasserstein-2
metric space. The flow moves rho toward rho_target along the geodesic of
minimal transport cost.

**JKO Scheme (Jordan-Kinderlehrer-Otto 1998)**

The continuous gradient flow is discretised via the proximal operator:

    rho_{k+1} = argmin_{rho} [ W_2(rho, rho_k)^2 / (2*tau) + F(rho) ]

Each JKO step moves rho one step along the Wasserstein gradient of F.
With F = KL(. || rho_target), the JKO scheme interpolates between rho_0
and rho_target along the W2 geodesic.

**Entropic Regularisation (Cuturi 2013)**

The exact W2 transport plan is expensive (O(N^3) linear program). We use
entropic regularisation (Sinkhorn algorithm):

    W_eps(mu, nu) = min_{gamma in Pi(mu,nu)} [<C, gamma> + eps * KL(gamma || mu x nu)]

Sinkhorn iterations (O(N^2) per step):
    u <- a / (K @ v)
    v <- b / (K.T @ u)
Where K_{ij} = exp(-C_{ij}/eps), a=mu weights, b=nu weights.

**Application to ETF Ranking**

For each ETF:
1. rho_0 = empirical distribution of log returns over rolling window
2. rho_target = macro-implied Gaussian N(mu_macro, sigma_macro)
3. Run JKO scheme for K steps: rho_0 -> rho_1 -> ... -> rho_K

Signals:
- W2(rho_0, rho_target): how far current distribution is from macro target
- sign(mean(rho_target) - mean(rho_0)): macro implies up or down move
- Flow velocity: speed of convergence toward target

**Distinction from OT-SIGNAL (in suite)**

OT-SIGNAL uses optimal transport to measure cross-sectional distances
between ETF return distributions (comparing ETFs to each other).
WPD uses the Wasserstein gradient flow (JKO scheme) to measure how far
each ETF's distribution is from a macro-implied TARGET — and how fast
it is moving toward that target. Completely different application of OT.

References
----------
- Jordan, R., Kinderlehrer, D. & Otto, F. (1998). The variational formulation
  of the Fokker-Planck equation. SIAM Journal on Mathematical Analysis, 29(1), 1–17.
- Villani, C. (2009). Optimal Transport: Old and New. Springer.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal
  transport. NeurIPS 2013.
- Peyre, G. & Cuturi, M. (2019). Computational optimal transport.
  Foundations and Trends in Machine Learning, 11(5-6), 355–607.
- Ambrosio, L., Gigli, N. & Savare, G. (2008). Gradient Flows in Metric
  Spaces and in the Space of Probability Measures. Birkhauser.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Sinkhorn algorithm ────────────────────────────────────────────────────────

def _sinkhorn(
    a:   np.ndarray,   # (N,) source weights
    b:   np.ndarray,   # (M,) target weights
    C:   np.ndarray,   # (N, M) cost matrix
    eps: float,
    n_iter: int,
) -> Tuple[np.ndarray, float]:
    """
    Sinkhorn-Knopp algorithm for entropic optimal transport.

    Returns
    -------
    P     : (N, M) regularised transport plan
    W_eps : scalar entropic W2 cost (Frobenius inner product <C, P>)
    """
    log_K = -C / eps
    # Numerical stabilisation: log-domain Sinkhorn
    log_u = np.zeros(len(a))
    log_v = np.zeros(len(b))
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)

    for _ in range(n_iter):
        # log_u = log_a - logsumexp(log_K + log_v)
        log_v_row = log_K + log_v[None, :]   # (N, M)
        lse_u = log_v_row.max(axis=1) + np.log(np.exp(log_v_row - log_v_row.max(axis=1, keepdims=True)).sum(axis=1))
        log_u = log_a - lse_u

        log_u_col = log_K + log_u[:, None]   # (N, M)
        lse_v = log_u_col.max(axis=0) + np.log(np.exp(log_u_col - log_u_col.max(axis=0, keepdims=True)).sum(axis=0))
        log_v = log_b - lse_v

    log_P = log_u[:, None] + log_K + log_v[None, :]
    P     = np.exp(log_P)
    W_eps = float(np.sum(C * P))
    return P, W_eps


# ── W2 distance (1D exact via sorting) ───────────────────────────────────────

def _w2_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Exact W2 distance between two 1D empirical distributions.
    W2^2 = mean((sort(x) - sort(y))^2) when len(x) == len(y).
    If different sizes, interpolate quantiles.
    """
    if len(x) == len(y):
        return float(np.sqrt(np.mean((np.sort(x) - np.sort(y))**2)))
    # Quantile interpolation
    q  = np.linspace(0, 1, 200)
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    return float(np.sqrt(np.mean((qx - qy)**2)))


# ── JKO proximal step ─────────────────────────────────────────────────────────

def _jko_step(
    rho_samples:    np.ndarray,   # (N,) current distribution samples
    target_samples: np.ndarray,   # (M,) target distribution samples
    tau:            float,
    eps:            float,
    n_iter:         int,
) -> np.ndarray:
    """
    One JKO proximal step: move rho one step toward target in W2 space.

    Implements the entropic JKO step:
        rho_{k+1} samples = weighted average of rho_k and optimal transport plan

    Approximation: use Sinkhorn to find the transport plan P, then the
    updated samples are the barycentre along the geodesic at time tau.

    x_new_i = (1 - tau) * x_i + tau * sum_j P_{ij}/a_i * y_j
    """
    N = len(rho_samples)
    M = len(target_samples)

    # Cost matrix: squared Euclidean distance
    C = (rho_samples[:, None] - target_samples[None, :])**2   # (N, M)

    # Uniform weights
    a = np.ones(N) / N
    b = np.ones(M) / M

    # Sinkhorn transport plan
    P, _ = _sinkhorn(a, b, C, eps, n_iter)

    # Barycentric projection: x_new_i = (1-tau)*x_i + tau * E[y | x_i]
    # E[y | x_i] = sum_j P_{ij}/a_i * y_j
    cond_mean = (P @ target_samples) / (a + 1e-300)   # (N,)
    x_new     = (1 - tau) * rho_samples + tau * cond_mean

    return x_new


# ── Macro-implied target distribution ────────────────────────────────────────

def _macro_target(
    log_ret:    np.ndarray,   # ETF log returns (for calibration)
    macro_vals: np.ndarray,   # latest macro signal values (normalised)
    n_samples:  int,
    rng:        np.random.Generator,
) -> np.ndarray:
    """
    Construct macro-implied target distribution as a Gaussian.

    mu_macro    = linear combination of macro signals -> implied drift
    sigma_macro = VIX-implied vol

    Parameters
    ----------
    log_ret    : recent ETF log returns (used to calibrate scale)
    macro_vals : normalised macro values [VIX, DXY, T10Y2Y, ...]
    n_samples  : number of target samples to draw
    rng        : random generator

    Returns
    -------
    target_samples : (n_samples,) samples from macro-implied distribution
    """
    # ETF empirical moments (for scale calibration)
    emp_std  = log_ret.std() + 1e-8

    # Macro-implied drift: use macro signals as directional inputs
    # VIX: negative (high VIX = negative expected return)
    # DXY: mixed (negative for commodities, neutral for equity)
    # T10Y2Y: positive (steepening = risk-on)
    M = len(macro_vals)
    weights = np.zeros(M)
    if M >= 1: weights[0] = -0.4  # VIX: negative
    if M >= 2: weights[1] = -0.2  # DXY: slightly negative
    if M >= 3: weights[2] =  0.4  # T10Y2Y: positive (risk-on)

    mu_macro    = float(np.dot(weights[:M], macro_vals[:M])) * config.MACRO_RETURN_SCALE
    # Clamp to reasonable return range
    mu_macro    = float(np.clip(mu_macro, -3 * emp_std, 3 * emp_std))

    # Macro-implied vol: use VIX level if available
    if M >= 1:
        vix_level   = max(macro_vals[0] * 15 + 20, 5)  # rough VIX level
        sigma_macro = (vix_level / 100) / np.sqrt(252)  # daily vol from VIX
    else:
        sigma_macro = emp_std

    # Draw samples from macro-implied N(mu_macro, sigma_macro)
    return rng.normal(mu_macro, sigma_macro, n_samples)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_wpd_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Compute Wasserstein Proximal Descent scores for all ETFs.

    For each ETF:
      1. rho_0 = empirical return distribution over rolling window
      2. rho_target = macro-implied Gaussian distribution
      3. Run JKO gradient flow: rho_0 -> ... -> rho_K
      4. Score from W2 geodesic distance and flow direction

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite WPD z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    if len(prices) < window + 5:
        return pd.Series(dtype=float)

    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    # Latest normalised macro values
    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu       = np.nanmean(macro_vals, axis=0)
        m_std      = np.nanstd(macro_vals,  axis=0) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
        macro_latest = macro_norm[-1]
    else:
        macro_latest = np.zeros(0)

    rng        = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        ps = prices_a[ticker].dropna()
        if len(ps) < window + 2:
            continue

        log_ret = np.log(ps / ps.shift(1)).dropna().values[-window:]

        if len(log_ret) < 10:
            continue

        # ── Source distribution: empirical returns ────────────────────────────
        N = min(config.N_SAMPLES, len(log_ret))
        idx_sample = rng.choice(len(log_ret), size=N, replace=(N > len(log_ret)))
        rho_0      = log_ret[idx_sample]

        # ── Target distribution: macro-implied Gaussian ───────────────────────
        rho_target = _macro_target(log_ret, macro_latest, N, rng)

        # ── W2 geodesic distance (exact 1D) ──────────────────────────────────
        w2_dist = _w2_1d(rho_0, rho_target)

        # ── JKO gradient flow ─────────────────────────────────────────────────
        rho_current = rho_0.copy()
        for step in range(config.JKO_N_STEPS):
            rho_current = _jko_step(
                rho_samples    = rho_current,
                target_samples = rho_target,
                tau            = config.JKO_TAU,
                eps            = config.SINKHORN_EPS,
                n_iter         = config.SINKHORN_ITER,
            )

        # ── Flow velocity: W2 moved per unit time ─────────────────────────────
        w2_after = _w2_1d(rho_current, rho_target)
        flow_vel = max(0.0, w2_dist - w2_after) / (config.JKO_N_STEPS * config.JKO_TAU + 1e-8)

        # ── Flow direction: does macro imply higher or lower returns? ─────────
        mean_target = float(np.mean(rho_target))
        mean_source = float(np.mean(rho_0))
        flow_dir    = float(np.sign(mean_target - mean_source))

        print(f"    {ticker}: W2={w2_dist:.5f}  dir={flow_dir:+.0f}  "
              f"vel={flow_vel:.5f}  target_mean={mean_target:.5f}")

        # ── Score construction ────────────────────────────────────────────────
        # Large W2 + positive direction → ETF has room to move up → positive
        # Large W2 + negative direction → ETF overpriced vs macro → negative
        s_geodesic  = w2_dist * flow_dir     # signed geodesic distance
        s_direction = flow_dir               # macro implied direction
        s_velocity  = flow_vel * flow_dir    # signed flow speed

        composite = (
            config.WEIGHT_GEODESIC  * s_geodesic
            + config.WEIGHT_DIRECTION * s_direction
            + config.WEIGHT_VELOCITY  * s_velocity
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
