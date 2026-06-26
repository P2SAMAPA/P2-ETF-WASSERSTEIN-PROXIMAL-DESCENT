# 🌊 P2-ETF-WASSERSTEIN-PROXIMAL-DESCENT

**Wasserstein Proximal Descent Engine — JKO Scheme (Jordan, Kinderlehrer & Otto 1998)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine models the evolution of each ETF's return distribution as a
**gradient flow in Wasserstein-2 space** toward a macro-implied target
distribution. The Wasserstein distance between the current empirical return
distribution and the macro-implied target — and the speed at which the
distribution is flowing toward that target — provide tradeable signals.

---

## Theory

### Wasserstein-2 Distance

```
W2(mu, nu)^2 = inf_{gamma in Pi(mu,nu)} integral |x-y|^2 d_gamma(x,y)
```

For 1D empirical distributions: W2 = L2 distance between sorted quantiles.

### Gradient Flow (Fokker-Planck)

```
d_rho/dt = div(rho * grad(log(rho / rho_target)))
```

This moves rho toward rho_target along the geodesic of minimal W2 cost —
the steepest descent of KL(rho || rho_target) in Wasserstein space.

### JKO Proximal Scheme (1998)

```
rho_{k+1} = argmin [ W2(rho, rho_k)^2 / (2*tau)  +  KL(rho || rho_target) ]
```

Each step moves rho one proximal step along the W2 gradient of KL divergence.

### Sinkhorn Algorithm (Cuturi 2013)

Each JKO step is solved via entropic regularisation:

```
K = exp(-C / eps)
u <- a / (K @ v),   v <- b / (K.T @ u)   [iterate until convergence]
P = diag(u) K diag(v)                     [transport plan]
```

O(N^2) per iteration, ~100 iterations — very fast for N=100 samples.

### Score Construction

```
score = 0.40 * W2 * direction  +  0.40 * direction  +  0.20 * velocity * direction
```

| Component | Meaning | Signal |
|-----------|---------|--------|
| W2 x direction | Signed geodesic distance to macro target | Large gap + upside = positive |
| Direction | sign(mean_target - mean_source) | Macro implies up or down |
| Velocity x direction | Signed flow speed toward target | Fast convergence + upside = positive |

---

## Distinction from OT-SIGNAL

| Engine | OT Application |
|--------|---------------|
| OT-SIGNAL | Cross-sectional W2 distances between ETF distributions |
| **WPD (this engine)** | W2 gradient flow of each ETF toward macro-implied target |

Completely different use of optimal transport — OT-SIGNAL compares ETFs to
each other; WPD compares each ETF to a macro-implied target distribution.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-WASSERSTEIN-PROXIMAL-DESCENT/
├── config.py          # Universes, JKO params, Sinkhorn eps, score weights
├── data_manager.py    # HuggingFace loader
├── wpd_engine.py      # Core: Sinkhorn, JKO step, W2 distance, scoring
├── trainer.py         # Orchestrator
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (Sinkhorn O(N^2) — very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-WASSERSTEIN-PROXIMAL-DESCENT
cd P2-ETF-WASSERSTEIN-PROXIMAL-DESCENT
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-wasserstein-proximal-results`

---

## References

- Jordan, R., Kinderlehrer, D. & Otto, F. (1998). The variational formulation
  of the Fokker-Planck equation. *SIAM Journal on Mathematical Analysis*, 29(1), 1–17.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal
  transport. *NeurIPS 2013*.
- Peyre, G. & Cuturi, M. (2019). Computational optimal transport.
  *Foundations and Trends in Machine Learning*, 11(5-6), 355–607.
- Ambrosio, L., Gigli, N. & Savare, G. (2008). *Gradient Flows in Metric
  Spaces and in the Space of Probability Measures*. Birkhauser.
