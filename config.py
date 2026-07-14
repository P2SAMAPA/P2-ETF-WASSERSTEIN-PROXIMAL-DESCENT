import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-wasserstein-proximal-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Empirical distribution construction ──────────────────────────────────────
# Each ETF's return distribution is represented as N_SAMPLES empirical samples
# drawn from the rolling window of log returns.
N_SAMPLES = 100   # number of empirical samples (subsampled if window > N_SAMPLES)

# ── Macro-implied target distribution ────────────────────────────────────────
# The target distribution rho_1 is constructed from a macro-conditioned
# Gaussian: N(mu_macro, sigma_macro) where:
#   mu_macro    = mean return implied by macro regime (VIX, DXY, T10Y2Y)
#   sigma_macro = vol implied by VIX level
# This gives the "macro-fair" return distribution for each ETF.
MACRO_RETURN_SCALE = 0.01    # scale factor for macro-implied drift
MACRO_VOL_SCALE    = 0.01    # scale factor: VIX / 100 -> annualised vol

# ── JKO scheme (Jordan-Kinderlehrer-Otto) ────────────────────────────────────
# WPD solves: rho_{k+1} = argmin_rho [W2(rho, rho_k)^2 / (2*tau) + F(rho)]
# where F(rho) = KL(rho || rho_target) is the free energy functional.
# Each JKO step is solved via entropic regularisation (Sinkhorn algorithm).
JKO_N_STEPS   = 5      # number of JKO gradient flow steps
JKO_TAU       = 0.5    # step size (time step in gradient flow)
SINKHORN_EPS  = 0.05   # entropic regularisation (lower = more accurate, slower)
SINKHORN_ITER = 100    # Sinkhorn iterations

# ── Score construction ────────────────────────────────────────────────────────
# After running the JKO gradient flow from rho_0 (empirical) toward rho_1 (target):
#
#   geodesic_dist  : W2(rho_0, rho_target) — how far current distribution
#                    is from macro-implied target
#                    Small = ETF already aligned with macro → neutral
#                    Large = ETF far from macro target → regime gap → signal
#
#   flow_direction : sign of (mean(rho_target) - mean(rho_0))
#                    Positive = target has higher mean → macro implies upside
#                    Negative = target has lower mean → macro implies downside
#
#   flow_velocity  : W2(rho_0, rho_1_after_JKO) / (JKO_N_STEPS * JKO_TAU)
#                    How fast the distribution is moving toward the target
#                    (the "speed" of the gradient flow)

WEIGHT_GEODESIC  = 0.40
WEIGHT_DIRECTION = 0.40
WEIGHT_VELOCITY  = 0.20

TOP_N = 3
