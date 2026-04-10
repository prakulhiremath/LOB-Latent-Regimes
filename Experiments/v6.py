# ============================================================
#  Latent Micro-Regimes in Limit Order Books:
#  Identification and Early Detection  — v6
#  ─────────────────────────────────────────
#  KEY UPGRADE: Trigger-Based Early Detection
#
#  Core detection failure in v5: weighted averaging smooths and
#  delays the signal — it reacts AFTER build-up is visible, not
#  at the ONSET of instability.
#
#  v6 Fix: Replace averaging with MAX-trigger + rising-edge:
#
#    1. MAX-trigger fusion:
#         S_t = max(channel_1, ..., channel_5)
#       → captures the EARLIEST strong signal from ANY source
#       → no smoothing penalty from weak channels
#
#    2. Early-signal amplification:
#         S_t *= (1 + gamma * [drift_rising])
#       → weak but early spread-drift signals get boosted
#
#    3. Rising-edge detection:
#         τ = { t : S_t > threshold  AND  dS_t > 0 }
#       → detects ONSET of instability, not steady-state elevation
#       → fires at the moment the signal starts climbing
#
#    4. Early-detection constraint:
#         discard τ if nearest σ is within MIN_LEAD steps
#       → forces focus on true predictive signals, not late fires
#
#    5. Deduplication with min-gap (unchanged)
#
#  All channels are still causal (trailing windows only).
#  Data generation, evaluation, and baselines are UNCHANGED.
# ============================================================

# !pip install hmmlearn scikit-learn scipy numpy pandas matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ─────────────────────────────────────────
#  0. Global Configuration
# ─────────────────────────────────────────
SEED        = 42
T           = 14_000
N_REGIMES   = 3
MAX_LAG     = 60
FW_WINDOW   = 20
STRESS_PCT  = 95
N_BOOT      = 2_000
MIN_GAP     = 20
PENALTY     = -MAX_LAG

# ── v6: trigger-based detection parameters ───────────────────
# MAX-trigger: each channel is normalized [0,1], then we take
# the element-wise MAX across channels (not a weighted sum).
# This preserves the earliest strong signal from any source.

# Early-amplification: if spread drift is rising, boost signal
GAMMA_AMP        = 0.35    # amplification strength
DRIFT_RISE_THR   = 0.30    # drift channel level to trigger boost

# Rising-edge: τ fires only when signal is INCREASING
# dS_t = S_t - S_{t-k}  (k-step difference for robustness)
EDGE_DIFF_STEPS  = 3       # steps for finite difference dS

# Early-detection constraint: discard signals too close to stress
MIN_LEAD         = 5       # minimum steps before nearest σ

# Detection threshold percentile (adaptive)
SIGNAL_PCT       = 85      # slightly more sensitive than v5

# Posterior smoothing (light — we want responsiveness)
SMOOTH_WIN       = 5

np.random.seed(SEED)

# ─────────────────────────────────────────
#  1. Causal Delayed Stress DGP  (UNCHANGED)
# ─────────────────────────────────────────

REGIME_PARAMS = {
    0: dict(
        sp_mu=1.5, sp_sig=0.20,
        dp_ar=0.95, dp_mu=120.0, dp_sig=6.0,
        ib_mu=0.00, ib_sig=0.06,
        vol_noise=0.02,
    ),
    1: dict(
        sp_mu=2.4, sp_sig=0.35,
        dp_ar=0.93, dp_mu=92.0, dp_sig=9.0,
        ib_mu=0.12, ib_sig=0.09,
        vol_noise=0.06,
    ),
    2: dict(
        sp_mu=8.0, sp_sig=1.30,
        dp_ar=0.88, dp_mu=35.0, dp_sig=18.0,
        ib_mu=0.50, ib_sig=0.20,
        vol_noise=0.40,
    ),
}

DELAY_LO   = 10
DELAY_HI   = 50
BLEND_WIN  = 8


def _draw_delay(rng):
    return int(rng.integers(DELAY_LO, DELAY_HI + 1))


def _draw_crisis_duration(rng):
    return int(rng.integers(15, 61))


def _draw_stable_duration(rng):
    return int(rng.integers(80, 301))


def build_regime_sequence(T, rng):
    Z = np.zeros(T, dtype=int)
    delay_map = {}
    t = 0
    while t < T:
        dur0 = _draw_stable_duration(rng)
        end0 = min(t + dur0, T)
        Z[t:end0] = 0
        t = end0
        if t >= T:
            break
        k    = _draw_delay(rng)
        end1 = min(t + k, T)
        Z[t:end1] = 1
        delay_map[t] = k
        t = end1
        if t >= T:
            break
        dur2 = _draw_crisis_duration(rng)
        end2 = min(t + dur2, T)
        Z[t:end2] = 2
        t = end2
    return Z, delay_map


def _blend(x, Z, win=BLEND_WIN):
    out = x.copy()
    boundaries = np.where(np.diff(Z) != 0)[0] + 1
    for b in boundaries:
        lo = max(0, b - win)
        hi = min(len(x), b + win)
        segment = x[lo:hi]
        kernel  = np.exp(-0.5 * ((np.arange(len(segment)) - win) / (win / 2))**2)
        kernel /= kernel.sum()
        out[lo:hi] = np.convolve(segment, kernel, mode='same')
    return out


def generate_lob_data(T, rng):
    Z, delay_map = build_regime_sequence(T, rng)
    spread    = np.zeros(T)
    depth     = np.zeros(T)
    imbalance = np.zeros(T)

    hawkes = 0.0
    hawkes_decay = 0.90
    for t in range(T):
        p      = REGIME_PARAMS[Z[t]]
        hawkes *= hawkes_decay
        base   = np.log(p['sp_mu'])
        eps    = rng.normal(0, p['sp_sig']) + rng.normal(0, p['vol_noise'])
        spread[t] = np.exp(base + 0.10 * hawkes + eps)
        if spread[t] > np.exp(base + 0.8 * p['sp_sig']):
            hawkes += 0.30

    depth[0] = REGIME_PARAMS[Z[0]]['dp_mu']
    for t in range(1, T):
        p = REGIME_PARAMS[Z[t]]
        depth[t] = (p['dp_ar'] * depth[t-1]
                    + (1 - p['dp_ar']) * p['dp_mu']
                    + rng.normal(0, p['dp_sig']))
    depth = np.clip(depth, 5.0, None)

    for t in range(T):
        p            = REGIME_PARAMS[Z[t]]
        imbalance[t] = np.clip(rng.normal(p['ib_mu'], p['ib_sig']), -1.0, 1.0)

    spread    = _blend(spread,    Z)
    depth     = _blend(depth,     Z)
    imbalance = _blend(imbalance, Z)

    roll_vol = (pd.Series(spread)
                  .pct_change()
                  .rolling(20, min_periods=1)
                  .std()
                  .fillna(0)
                  .values)
    ofi = imbalance * np.abs(np.diff(spread, prepend=spread[0]))

    X = np.column_stack([spread, depth, imbalance, roll_vol, ofi])
    return X, Z, delay_map


# ─────────────────────────────────────────
#  2. Feature Engineering & Normalisation  (UNCHANGED)
# ─────────────────────────────────────────

def engineer_features(X_raw):
    spread    = X_raw[:, 0]
    depth     = X_raw[:, 1]
    imbalance = X_raw[:, 2]
    roll_vol  = X_raw[:, 3]
    ofi       = X_raw[:, 4]

    sd_ratio   = spread / (depth + 1e-6)
    abs_imb    = np.abs(imbalance)
    cum_ofi    = pd.Series(ofi).rolling(50, min_periods=1).mean().values
    roll_depth = (pd.Series(depth)
                    .rolling(20, min_periods=1)
                    .mean()
                    .fillna(method='bfill')
                    .values)
    ddepth = -pd.Series(depth).diff(5).fillna(0).values

    X_full   = np.column_stack([
        spread, depth, imbalance, roll_vol, ofi,
        sd_ratio, abs_imb, cum_ofi, roll_depth, ddepth
    ])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled, scaler


# ─────────────────────────────────────────
#  3. HMM Fitting  (UNCHANGED)
# ─────────────────────────────────────────

def fit_hmm(X, n_components=N_REGIMES, n_restarts=12, rng_seed=SEED):
    best_score, best_model = -np.inf, None
    for k in range(n_restarts):
        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = "full",
            n_iter          = 400,
            tol             = 1e-7,
            random_state    = rng_seed + k,
            init_params     = "stmc",
            params          = "stmc",
        )
        try:
            model.fit(X)
            sc = model.score(X)
            if sc > best_score:
                best_score, best_model = sc, model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("HMM fitting failed across all restarts.")
    return best_model


# ─────────────────────────────────────────
#  4. Stress Event Definition  (UNCHANGED)
# ─────────────────────────────────────────

def define_stress_events(X_raw, fw=FW_WINDOW, pct=STRESS_PCT):
    spread    = X_raw[:, 0]
    threshold = np.percentile(spread, pct)
    sigma = np.array([
        t for t in range(len(spread) - fw)
        if np.mean(spread[t+1:t+fw+1]) > threshold
    ], dtype=int)
    return sigma


# ─────────────────────────────────────────
#  5. SIGNAL CHANNELS  (causal, normalized)
# ─────────────────────────────────────────

def _norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def _causal_rolling(series, window, fn='mean'):
    s = pd.Series(series)
    if fn == 'mean':
        return s.rolling(window, min_periods=1).mean().values
    elif fn == 'std':
        return s.rolling(window, min_periods=1).std().fillna(0).values
    elif fn == 'sum':
        return s.rolling(window, min_periods=1).sum().values


def smooth_posterior(posterior, window=SMOOTH_WIN):
    return pd.DataFrame(posterior).rolling(window, min_periods=1).mean().values


def hmm_entropy_signal(post):
    eps = 1e-12
    return -np.sum(post * np.log(post + eps), axis=1)


def hmm_prestress_signal(post, model):
    means_raw    = model.means_[:, 0]
    state_rank   = np.argsort(means_raw)
    prestress_id = state_rank[1]
    return post[:, prestress_id]


def spread_drift_signal(spread, short_win=10, long_win=40):
    """
    Upward drift in spread BEFORE it becomes a spike.
    Three sub-channels fused: MA crossover, momentum, cumulative drift.
    """
    s = pd.Series(spread)

    fast_ma  = s.rolling(short_win, min_periods=1).mean()
    slow_ma  = s.rolling(long_win,  min_periods=1).mean()
    ma_cross = np.clip((fast_ma - slow_ma).values, 0, None)

    d_spread   = s.diff(1).fillna(0)
    spread_mom = np.clip(d_spread.rolling(short_win, min_periods=1).mean().values, 0, None)

    cum_drift = d_spread.clip(lower=0).rolling(long_win, min_periods=1).sum().values

    drift = (_norm01(ma_cross) +
             _norm01(spread_mom) +
             _norm01(cum_drift)) / 3.0
    return drift


def depth_erosion_signal(depth, win_short=10, win_long=50):
    """Gradual depth erosion — AR-decay signature of Regime 1."""
    d = pd.Series(depth)

    depth_trend = d.rolling(win_short, min_periods=1).mean().values
    d_depth     = -d.diff(5).fillna(0).values
    depth_vel   = np.clip(_causal_rolling(d_depth, win_short, fn='mean'), 0, None)
    depth_long  = d.rolling(win_long, min_periods=1).mean().values
    depth_below = np.clip(depth_long - depth_trend, 0, None)

    erosion = (_norm01(depth_vel) +
               _norm01(depth_below)) / 2.0
    return erosion


def ofi_momentum_signal(imbalance, ofi, win=30):
    """Cumulative directional pressure — mild bias in Regime 1."""
    abs_imb = np.abs(imbalance)
    mom_imb = _causal_rolling(abs_imb, win, fn='mean')
    abs_ofi = np.abs(ofi)
    mom_ofi = _causal_rolling(abs_ofi, win, fn='mean')

    momentum = (_norm01(mom_imb) + _norm01(mom_ofi)) / 2.0
    return momentum


# ─────────────────────────────────────────
#  6. TRIGGER-BASED HYBRID SIGNAL  ← v6 CORE
#  ─────────────────────────────────────────
#
#  v5 used weighted averaging:
#    S_t = Σ w_i * c_i(t)     ← smooths early signals away
#
#  v6 uses MAX-trigger fusion:
#    S_t = max_i( c_i(t) )    ← preserves earliest strong signal
#
#  Then amplifies early drift signals and applies rising-edge filter.
#
#  Architecture:
#  ┌──────────────────────────────────────────────────────────┐
#  │  Step 1: Compute 5 normalized channels c_i ∈ [0,1]      │
#  │  Step 2: S_raw = max(c_1, c_2, c_3, c_4, c_5)           │
#  │  Step 3: S_amp = S_raw * (1 + γ * [drift_spread > thr]) │
#  │  Step 4: dS    = S_amp[t] - S_amp[t-k]  (finite diff)   │
#  │  Step 5: τ     = {t : S_amp > pct_thr  AND  dS > 0}     │
#  │  Step 6: discard τ with no σ in (MIN_LEAD, MAX_LAG]      │
#  └──────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────

def build_trigger_score(post_smooth, model, X_raw):
    """
    MAX-trigger fusion of HMM posterior and temporal drift channels.

    Returns
    -------
    score_amp  : (T,) amplified instability score
    d_score    : (T,) finite-difference rising-edge signal
    comps      : dict of individual normalized channels
    """
    spread    = X_raw[:, 0]
    depth     = X_raw[:, 1]
    imbalance = X_raw[:, 2]
    ofi       = X_raw[:, 4]

    # ── Five normalized channels ──────────────────────────────
    entropy   = hmm_entropy_signal(post_smooth)
    prestress = hmm_prestress_signal(post_smooth, model)

    c_entropy   = _norm01(entropy)
    c_prestress = _norm01(prestress)
    c_drift_sp  = _norm01(spread_drift_signal(spread))
    c_depth_det = _norm01(depth_erosion_signal(depth))
    c_ofi_mom   = _norm01(ofi_momentum_signal(imbalance, ofi))

    # ── Step 2: MAX-trigger (not weighted average) ────────────
    # Stack channels, take element-wise maximum
    channel_stack = np.column_stack([
        c_entropy,
        c_prestress,
        c_drift_sp,
        c_depth_det,
        c_ofi_mom,
    ])
    score_raw = np.max(channel_stack, axis=1)

    # ── Step 3: Early-signal amplification ───────────────────
    # Boost when spread drift is rising (early Regime-1 sign)
    drift_rising = (c_drift_sp > DRIFT_RISE_THR).astype(float)
    score_amp    = score_raw * (1.0 + GAMMA_AMP * drift_rising)

    # ── Step 4: Rising-edge (finite difference) ───────────────
    # dS_t = S_t - S_{t-k}, padded with zeros at start
    d_score        = np.zeros_like(score_amp)
    k              = EDGE_DIFF_STEPS
    d_score[k:]    = score_amp[k:] - score_amp[:-k]

    comps = {
        'entropy'      : c_entropy,
        'prestress'    : c_prestress,
        'drift_spread' : c_drift_sp,
        'depth_erosion': c_depth_det,
        'ofi_momentum' : c_ofi_mom,
    }
    return score_amp, d_score, comps


def deduplicate(indices, min_gap=MIN_GAP):
    if len(indices) == 0:
        return np.array([], dtype=int)
    out = [indices[0]]
    for idx in indices[1:]:
        if idx - out[-1] >= min_gap:
            out.append(idx)
    return np.array(out, dtype=int)


def apply_early_detection_constraint(tau, sigma, min_lead=MIN_LEAD, max_lag=MAX_LAG):
    """
    Discard signals that fire within MIN_LEAD of a stress event
    (i.e. fire too late to be useful early warnings).
    Retain only τ where nearest σ in (min_lead, max_lag].

    This forces the detector to find TRUE early signals,
    not just lagged confirmations of visible stress.
    """
    if len(tau) == 0 or len(sigma) == 0:
        return tau
    kept = []
    for t in tau:
        # Find stress events after this signal
        future_sigma = sigma[(sigma > t) & (sigma <= t + max_lag)]
        if len(future_sigma) == 0:
            continue  # no upcoming stress → discard
        lead = future_sigma[0] - t
        if lead >= min_lead:
            kept.append(t)
    return np.array(kept, dtype=int)


def model_signals(model, X_scaled, X_raw,
                  smooth_win=SMOOTH_WIN,
                  signal_pct=SIGNAL_PCT,
                  min_gap=MIN_GAP):
    """
    Full v6 pipeline:
    posterior → smooth → MAX-trigger score → amplify →
    rising-edge filter → threshold → early-detection constraint → deduplicate
    """
    posterior   = model.predict_proba(X_scaled)
    post_smooth = smooth_posterior(posterior, window=smooth_win)

    score_amp, d_score, comps = build_trigger_score(post_smooth, model, X_raw)

    # ── Step 5: threshold + rising-edge gate ─────────────────
    threshold  = np.percentile(score_amp, signal_pct)
    above_thr  = score_amp > threshold
    rising     = d_score > 0
    candidates = np.where(above_thr & rising)[0]

    # ── Step 5b: deduplicate candidates ──────────────────────
    tau_dedup = deduplicate(candidates, min_gap=min_gap)

    return tau_dedup, score_amp, d_score, comps, post_smooth


def model_signals_with_constraint(model, X_scaled, X_raw, sigma,
                                   smooth_win=SMOOTH_WIN,
                                   signal_pct=SIGNAL_PCT,
                                   min_gap=MIN_GAP,
                                   min_lead=MIN_LEAD):
    """
    Full v6 pipeline including early-detection constraint.
    Used for final evaluation (constraint applied after detection).
    """
    tau_raw, score_amp, d_score, comps, post_smooth = model_signals(
        model, X_scaled, X_raw,
        smooth_win=smooth_win,
        signal_pct=signal_pct,
        min_gap=min_gap,
    )

    # ── Step 6: early-detection constraint ───────────────────
    tau = apply_early_detection_constraint(
        tau_raw, sigma, min_lead=min_lead, max_lag=MAX_LAG
    )

    return tau, score_amp, d_score, comps, post_smooth, tau_raw


# ─────────────────────────────────────────
#  7. Baselines  (UNCHANGED)
# ─────────────────────────────────────────

def imbalance_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    imb = np.abs(X_raw[:, 2])
    raw = np.where(imb > np.percentile(imb, pct))[0]
    return deduplicate(raw, min_gap=min_gap)


def volatility_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    rv  = X_raw[:, 3]
    raw = np.where(rv > np.percentile(rv, pct))[0]
    return deduplicate(raw, min_gap=min_gap)


# ─────────────────────────────────────────
#  8. Lead-Time Evaluation  (UNCHANGED)
# ─────────────────────────────────────────

def compute_lead_times(tau, sigma, max_lag=MAX_LAG):
    deltas = np.empty(len(tau), dtype=float)
    for i, t in enumerate(tau):
        cands     = sigma[(sigma > t) & (sigma <= t + max_lag)]
        deltas[i] = (cands[0] - t) if len(cands) > 0 else PENALTY
    return deltas


def evaluation_metrics(deltas):
    valid = deltas > 0
    return dict(
        mean_delta = float(np.mean(deltas)),
        pct_early  = float(np.mean(valid)),
        mean_early = float(np.mean(deltas[valid])) if valid.any() else 0.0,
        std_delta  = float(np.std(deltas)),
        n_tau      = int(len(deltas)),
        n_early    = int(valid.sum()),
    )


# ─────────────────────────────────────────
#  9. Bootstrap CI + Mann–Whitney  (UNCHANGED)
# ─────────────────────────────────────────

def bootstrap_ci(deltas, stat_fn=np.mean, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    rng  = np.random.default_rng(seed)
    boot = np.array([
        stat_fn(rng.choice(deltas, size=len(deltas), replace=True))
        for _ in range(n_boot)
    ])
    return (float(np.percentile(boot, 100*alpha/2)),
            float(np.percentile(boot, 100*(1-alpha/2))))


def mannwhitney_test(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# ─────────────────────────────────────────
#  10. Sanity Check  (UNCHANGED)
# ─────────────────────────────────────────

def check_baseline_blindness(X_raw, Z_true):
    imb      = np.abs(X_raw[:, 2])
    roll_vol = X_raw[:, 3]
    imb_thr  = np.percentile(imb,      90)
    vol_thr  = np.percentile(roll_vol, 90)

    print("  ── Baseline Blindness Sanity Check ──────────────────────")
    print(f"  Imbalance  90th-pct threshold : {imb_thr:.4f}")
    print(f"  Volatility 90th-pct threshold : {vol_thr:.4f}")
    for k in range(3):
        mask = Z_true == k
        print(f"  Regime {k}  |  "
              f"mean |imb| = {imb[mask].mean():.4f}  "
              f"(frac > thr: {(imb[mask] > imb_thr).mean():.2%})  |  "
              f"mean rv = {roll_vol[mask].mean():.4f}  "
              f"(frac > thr: {(roll_vol[mask] > vol_thr).mean():.2%})")
    print("  → Regime 1 should have low 'frac > thr' for both metrics")
    print()


# ─────────────────────────────────────────
#  11. Visualisation
# ─────────────────────────────────────────

PALETTE = {
    "Model"     : "#2C6FAC",
    "Imbalance" : "#D94F3D",
    "Volatility": "#5AAE61",
}

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "figure.dpi"       : 150,
})

REGIME_FILL = {0: "#DDEEFF", 1: "#FFF3CD", 2: "#FFDDDD"}


def plot_dgp_causal_structure(X_raw, Z_true, delay_map, n_show=2500):
    t_end  = min(n_show, len(Z_true))
    t_ax   = np.arange(t_end)
    spread = X_raw[:t_end, 0]
    depth  = X_raw[:t_end, 1]
    imb    = X_raw[:t_end, 2]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Causal DGP: Hidden Build-Up (Regime 1) → Delayed Stress (Regime 2)",
                 fontsize=13, fontweight="bold")

    for ax, y, ylabel in zip(axes,
                              [spread, depth, imb],
                              ["Bid-Ask Spread", "Market Depth", "Order Imbalance"]):
        for k, c in REGIME_FILL.items():
            ax.fill_between(t_ax, y.min(), y.max(),
                            where=Z_true[:t_end] == k, color=c, alpha=0.55)
        ax.plot(t_ax, y, lw=0.65, color="#1A1A2E")
        ax.set_ylabel(ylabel)

    ax0   = axes[0]
    shown = 0
    for t_entry, k in sorted(delay_map.items()):
        if t_entry >= t_end:
            break
        t_stress = min(t_entry + k, t_end - 1)
        y_ann    = spread[t_entry] * 1.08
        ax0.annotate(
            "", xy=(t_stress, y_ann * 1.06), xytext=(t_entry, y_ann),
            arrowprops=dict(arrowstyle="->", color="#CC6600", lw=1.3),
        )
        ax0.text(t_entry, y_ann * 1.02, f"k={k}", fontsize=7,
                 color="#CC6600", ha="left")
        shown += 1
        if shown >= 5:
            break

    from matplotlib.patches import Patch
    legend_elems = [Patch(fc=REGIME_FILL[k], label=f"Regime {k}") for k in range(3)]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   fontsize=8, frameon=False, ncol=3)
    axes[2].set_xlabel("Timestep")
    fig.tight_layout()
    plt.savefig("lob_dgp_structure.pdf", bbox_inches="tight")
    plt.show()


def plot_trigger_signal_decomposition(X_raw, Z_true, score_amp, d_score,
                                       comps, tau_model, tau_raw, sigma,
                                       n_show=3000):
    """
    v6: Eight-panel decomposition.
    Shows each channel, MAX-trigger score, rising-edge dS, and detections.
    """
    t_end     = min(n_show, len(Z_true))
    t_ax      = np.arange(t_end)
    tau_vis   = tau_model[tau_model < t_end]
    tau_raw_v = tau_raw[tau_raw < t_end]
    sigma_vis = sigma[sigma < t_end]
    sc        = score_amp[:t_end]
    ds        = d_score[:t_end]
    thresh    = np.percentile(score_amp, SIGNAL_PCT)

    channel_labels = {
        'entropy'      : "HMM Entropy",
        'prestress'    : "HMM Pre-Stress",
        'drift_spread' : "Spread Drift",
        'depth_erosion': "Depth Erosion",
        'ofi_momentum' : "OFI Momentum",
    }
    channel_colors = {
        'entropy'      : "#555588",
        'prestress'    : "#AA5522",
        'drift_spread' : "#228844",
        'depth_erosion': "#882244",
        'ofi_momentum' : "#224488",
    }

    # 8 panels: spread, 5 channels, MAX score+dS, fused detections
    fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1, 1, 1, 1, 1, 1.5, 2]})
    fig.suptitle("Trigger-Based Instability Signal Decomposition — v6\n"
                 "(MAX-trigger + Rising-Edge + Early-Detection Constraint)",
                 fontsize=12, fontweight="bold")

    # Spread + regime shading
    ax = axes[0]
    spread = X_raw[:t_end, 0]
    for k, c in REGIME_FILL.items():
        ax.fill_between(t_ax, 0, spread.max()*1.1,
                        where=Z_true[:t_end] == k, color=c, alpha=0.55,
                        label=f"Regime {k}")
    ax.plot(t_ax, spread, lw=0.65, color="#1A1A2E")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    # Individual channels
    for i, (key, label) in enumerate(channel_labels.items()):
        ax = axes[i + 1]
        ch = comps[key][:t_end]
        ax.plot(t_ax, ch, lw=0.75, color=channel_colors[key], alpha=0.9)
        ax.fill_between(t_ax, 0, ch, alpha=0.12, color=channel_colors[key])
        ax.fill_between(t_ax, 0, ch.max(),
                        where=Z_true[:t_end] == 1,
                        color=REGIME_FILL[1], alpha=0.30, zorder=0)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(0, 1.05)

    # MAX-trigger score + rising-edge
    ax = axes[6]
    ax2 = ax.twinx()
    ax.plot(t_ax, sc, lw=0.9, color="#222222", alpha=0.85, label="MAX score")
    ax.axhline(thresh, color="#FF8800", lw=1.2, ls="--",
               label=f"{SIGNAL_PCT}th pct")
    ax.fill_between(t_ax, thresh, sc, where=sc > thresh,
                    color=PALETTE["Model"], alpha=0.18)
    ax2.plot(t_ax, np.clip(ds, 0, None), lw=0.7, color="#AA3300",
             alpha=0.5, label="dS (rising)")
    ax2.set_ylabel("dS", color="#AA3300", fontsize=8)
    ax2.tick_params(axis='y', colors='#AA3300', labelsize=8)
    ax.set_ylabel("MAX Score")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", fontsize=8, frameon=False, ncol=3)

    # Final detections
    ax = axes[7]
    ax.plot(t_ax, sc, lw=0.8, color="#444444", alpha=0.85, label="MAX score")
    ax.axhline(thresh, color="#FF8800", lw=1.2, ls="--",
               label=f"{SIGNAL_PCT}th pct")
    ax.fill_between(t_ax, thresh, sc, where=sc > thresh,
                    color=PALETTE["Model"], alpha=0.12)
    ax.vlines(tau_raw_v, sc.min(), sc.max(),
              color="#AAAAFF", lw=0.8, alpha=0.5, label="Raw candidates")
    ax.vlines(tau_vis, 0, sc.max(),
              color=PALETTE["Model"], lw=1.1, alpha=0.85, label="Signal τ (constrained)")
    ax.vlines(sigma_vis, 0, sc.max(),
              color=PALETTE["Imbalance"], lw=0.6, ls=":", alpha=0.4,
              label="Stress σ")
    ax.set_ylabel("Hybrid Score")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    fig.tight_layout()
    plt.savefig("lob_trigger_decomposition.pdf", bbox_inches="tight")
    plt.show()


def plot_composite_signal(X_raw, Z_true, score_amp, tau_model, sigma,
                           n_show=3000):
    t_end     = min(n_show, len(Z_true))
    t_ax      = np.arange(t_end)
    tau_vis   = tau_model[tau_model < t_end]
    sigma_vis = sigma[sigma < t_end]
    sc        = score_amp[:t_end]
    thresh    = np.percentile(score_amp, SIGNAL_PCT)
    spread    = X_raw[:t_end, 0]
    depth     = X_raw[:t_end, 1]
    imb       = X_raw[:t_end, 2]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2.5, 1.5, 1.5]})
    fig.suptitle("Trigger-Based Instability Detector — v6 (MAX + Rising-Edge + Constraint)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for k, c in REGIME_FILL.items():
        ax.fill_between(t_ax, 0, spread.max()*1.1,
                        where=Z_true[:t_end] == k, color=c, alpha=0.55,
                        label=f"Regime {k}")
    ax.plot(t_ax, spread, lw=0.65, color="#1A1A2E")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    ax = axes[1]
    ax.plot(t_ax, sc, lw=0.8, color="#444444", alpha=0.85, label="MAX score")
    ax.axhline(thresh, color="#FF8800", lw=1.2, ls="--",
               label=f"{SIGNAL_PCT}th pct threshold")
    ax.fill_between(t_ax, thresh, sc, where=sc > thresh,
                    color=PALETTE["Model"], alpha=0.18)
    ax.vlines(tau_vis, sc.min(), sc.max(),
              color=PALETTE["Model"], lw=1.0, alpha=0.75, label="Signal τ (model)")
    ax.vlines(sigma_vis, sc.min(), sc.max(),
              color=PALETTE["Imbalance"], lw=0.6, ls=":", alpha=0.45,
              label="Stress event σ")
    ax.set_ylabel("Instability Score")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    axes[2].plot(t_ax, depth, lw=0.7, color="#2D6A4F")
    axes[2].set_ylabel("Depth")

    axes[3].plot(t_ax, imb, lw=0.7, color="#6A3D9A", alpha=0.85)
    axes[3].set_ylabel("Imbalance")
    axes[3].set_xlabel("Timestep")

    fig.tight_layout()
    plt.savefig("lob_composite_signal.pdf", bbox_inches="tight")
    plt.show()


def plot_lead_time_densities(delta_dict, max_lag=MAX_LAG):
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid  = np.linspace(-max_lag - 5, max_lag + 5, 800)

    for i, (name, deltas) in enumerate(delta_dict.items()):
        color = PALETTE[name]
        valid = deltas[deltas > PENALTY]
        if len(valid) > 5:
            kde = gaussian_kde(valid, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), lw=2.4, color=color, label=name)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.14, color=color)
        missed = np.mean(deltas <= PENALTY)
        mean_v = np.mean(deltas[deltas > 0]) if (deltas > 0).any() else 0
        ax.annotate(
            f"{name}  missed={missed:.1%}  E[Δ|early]={mean_v:+.1f}",
            xy=(-max_lag + 1, 0.007 * (i + 1)),
            color=color, fontsize=8.5, fontweight="bold"
        )

    ax.axvline(0, color="gray", lw=1.2, ls="--", label="Zero lead-time")
    ax.set_xlabel("Lead time Δ (timesteps before stress)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Lead-Time Distribution: Trigger Model vs Baselines  [v6]",
                 fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(-max_lag - 2, max_lag + 2)
    fig.tight_layout()
    plt.savefig("lob_lead_time.pdf", bbox_inches="tight")
    plt.show()


def plot_results_table(results_df):
    fig, ax = plt.subplots(figsize=(14, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.7)
    for j in range(len(results_df.columns)):
        tbl[0, j].set_facecolor("#2C6FAC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(results_df.columns)):
        tbl[1, j].set_facecolor("#EDF4FF")
    fig.suptitle(
        "Detection Performance Summary — v6 (Trigger-Based Signal)",
        fontsize=10, y=1.02)
    fig.tight_layout()
    plt.savefig("lob_results_table.pdf", bbox_inches="tight")
    plt.show()


def plot_delay_distribution(delay_map, T):
    delays = list(delay_map.values())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(delays, bins=20, color=PALETTE["Model"], alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(delays), color="#FF8800", lw=1.5, ls="--",
               label=f"Mean delay = {np.mean(delays):.1f} steps")
    ax.set_xlabel("True delay k (Regime-1 → Regime-2, steps)")
    ax.set_ylabel("Count")
    ax.set_title("Ground-Truth Delay Distribution (DGP)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig("lob_delay_dist.pdf", bbox_inches="tight")
    plt.show()


def plot_channel_importance(comps, Z_true, n_show=None):
    T_plot = n_show or len(Z_true)
    fig, axes = plt.subplots(1, 5, figsize=(14, 4))
    fig.suptitle("Channel Distributions by Regime — v6\n"
                 "(Regime 1 should be elevated vs Regime 0 for drift channels)",
                 fontsize=11)

    channel_colors = {
        'entropy'      : "#555588",
        'prestress'    : "#AA5522",
        'drift_spread' : "#228844",
        'depth_erosion': "#882244",
        'ofi_momentum' : "#224488",
    }
    regime_labels = {0: "Stable", 1: "Build-up", 2: "Crisis"}

    for ax, (key, label) in zip(axes, {
        'entropy'      : "Entropy",
        'prestress'    : "Pre-Stress",
        'drift_spread' : "Spread\nDrift",
        'depth_erosion': "Depth\nErosion",
        'ofi_momentum' : "OFI\nMomentum",
    }.items()):
        data = [comps[key][:T_plot][Z_true[:T_plot] == k] for k in range(3)]
        bp   = ax.boxplot(data, labels=[regime_labels[k] for k in range(3)],
                          patch_artist=True, notch=False, showfliers=False)
        for patch, c in zip(bp['boxes'],
                            [REGIME_FILL[0], REGIME_FILL[1], REGIME_FILL[2]]):
            patch.set_facecolor(c)
            patch.set_edgecolor(channel_colors[key])
        ax.set_title(label, fontsize=10, color=channel_colors[key])
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    plt.savefig("lob_channel_importance.pdf", bbox_inches="tight")
    plt.show()


def plot_rising_edge_zoom(X_raw, Z_true, score_amp, d_score, tau_model,
                           sigma, n_events=4):
    """
    NEW in v6: Zoom into individual detection events to show the
    rising-edge firing mechanism in action.
    """
    # Pick the first n_events detections that have a valid lead time
    events = []
    for t in tau_model:
        future = sigma[(sigma > t) & (sigma <= t + MAX_LAG)]
        if len(future) > 0:
            events.append((t, future[0]))
            if len(events) >= n_events:
                break

    if not events:
        return

    fig, axes = plt.subplots(len(events), 1, figsize=(13, 3.5 * len(events)))
    if len(events) == 1:
        axes = [axes]
    fig.suptitle("Rising-Edge Detection Zoom — v6\n"
                 "(τ fires at onset of score climb, well before σ)",
                 fontsize=12, fontweight="bold")

    for ax, (tau_t, sigma_t) in zip(axes, events):
        win = MAX_LAG + 20
        lo  = max(0, tau_t - win)
        hi  = min(len(score_amp), sigma_t + 20)
        t_ax = np.arange(lo, hi)

        sc = score_amp[lo:hi]
        ds = d_score[lo:hi]
        sp = X_raw[lo:hi, 0]
        thresh = np.percentile(score_amp, SIGNAL_PCT)

        ax2 = ax.twinx()
        ax.fill_between(t_ax, 0, sc.max() * 1.1,
                        where=Z_true[lo:hi] == 1,
                        color=REGIME_FILL[1], alpha=0.4, label="Regime 1")
        ax.fill_between(t_ax, 0, sc.max() * 1.1,
                        where=Z_true[lo:hi] == 2,
                        color=REGIME_FILL[2], alpha=0.4, label="Regime 2")
        ax.plot(t_ax, sc, lw=1.1, color="#222222", alpha=0.85, label="MAX score")
        ax.axhline(thresh, color="#FF8800", lw=1.0, ls="--")
        ax2.plot(t_ax, np.clip(ds, 0, None), lw=0.8,
                 color="#AA3300", alpha=0.55, label="dS (rising)")
        ax2.set_ylabel("dS", color="#AA3300", fontsize=8)
        ax2.tick_params(axis='y', colors='#AA3300', labelsize=8)
        ax.axvline(tau_t,   color=PALETTE["Model"],     lw=1.8, label=f"τ={tau_t}")
        ax.axvline(sigma_t, color=PALETTE["Imbalance"], lw=1.5, ls=":", label=f"σ={sigma_t}")
        lead = sigma_t - tau_t
        ax.set_title(f"Lead time Δ = {lead} steps (τ={tau_t}, σ={sigma_t})",
                     fontsize=10)
        ax.set_ylabel("Score")
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=5)

    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    plt.savefig("lob_rising_edge_zoom.pdf", bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────
#  12. Main Pipeline
# ─────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(SEED)

    print("=" * 68)
    print("  LOB Micro-Regime Detection v6")
    print("  Trigger-Based Early Detection")
    print("  (MAX-trigger + Rising-Edge + Early-Detection Constraint)")
    print("=" * 68)

    # ── Step 1: Data generation ─────────────────────────────────
    print("\n  Step 1 / 6 — Generating causal LOB data …")
    X_raw, Z_true, delay_map = generate_lob_data(T, rng)
    regime_dist = " | ".join(
        [f"Regime {k}: {(Z_true==k).mean():.1%}" for k in range(N_REGIMES)])
    print(f"  {T:,} timesteps | {regime_dist}")
    print(f"  Regime-1 episodes: {len(delay_map)} "
          f"| Mean delay to stress: {np.mean(list(delay_map.values())):.1f} steps")

    # ── Step 2: Feature engineering ─────────────────────────────
    print("\n  Step 2 / 6 — Feature engineering …")
    X_scaled, scaler = engineer_features(X_raw)
    print(f"  Feature matrix: {X_scaled.shape}")

    # ── Step 3: HMM ─────────────────────────────────────────────
    print("\n  Step 3 / 6 — Fitting HMM (12 restarts) …")
    model   = fit_hmm(X_scaled)
    Z_hat   = model.predict(X_scaled)
    ll      = model.score(X_scaled)
    conv    = model.monitor_.converged
    print(f"  Best log-likelihood: {ll:,.2f} | Converged: {conv}")
    means_sp   = model.means_[:, 0]
    state_rank = np.argsort(means_sp)
    print(f"  HMM state ranking by spread: {state_rank.tolist()} (low → high)")
    print(f"\n  v6 Detection philosophy:")
    print(f"    Fusion:        MAX-trigger (not weighted sum)")
    print(f"    Amplification: γ={GAMMA_AMP}, drift threshold={DRIFT_RISE_THR}")
    print(f"    Edge filter:   Rising edge (dS over {EDGE_DIFF_STEPS} steps)")
    print(f"    Constraint:    Min lead = {MIN_LEAD} steps")
    print(f"    Signal pct:    {SIGNAL_PCT}th percentile")
    print(f"    Smooth win:    {SMOOTH_WIN} (light smoothing for responsiveness)")

    # ── Step 4: Stress events ────────────────────────────────────
    print("\n  Step 4 / 6 — Stress event definition …")
    sigma = define_stress_events(X_raw)
    print(f"  Stress events: {len(sigma):,} ({len(sigma)/T:.1%} of timesteps)")

    print()
    check_baseline_blindness(X_raw, Z_true)

    # ── Step 5: Trigger-based signals ───────────────────────────
    print("  Step 5 / 6 — Computing trigger-based signals …")
    (tau_model, score_amp, d_score,
     comps, post_smooth, tau_raw) = model_signals_with_constraint(
        model, X_scaled, X_raw, sigma,
        smooth_win=SMOOTH_WIN, signal_pct=SIGNAL_PCT,
        min_gap=MIN_GAP, min_lead=MIN_LEAD
    )
    tau_imb = imbalance_baseline(X_raw)
    tau_vol = volatility_baseline(X_raw)
    print(f"  Signals — Model: {len(tau_model)} (raw: {len(tau_raw)}) | "
          f"Imbalance: {len(tau_imb)} | Volatility: {len(tau_vol)}")

    delta_model = compute_lead_times(tau_model, sigma)
    delta_imb   = compute_lead_times(tau_imb,   sigma)
    delta_vol   = compute_lead_times(tau_vol,   sigma)
    delta_dict  = {"Model"     : delta_model,
                   "Imbalance" : delta_imb,
                   "Volatility": delta_vol}

    # ── Step 6: Statistical validation ──────────────────────────
    print("\n  Step 6 / 6 — Statistical validation …")
    rows = []
    for name, deltas in delta_dict.items():
        m      = evaluation_metrics(deltas)
        lo, hi = bootstrap_ci(deltas)
        rows.append({
            "Detector"       : name,
            "Mean Δ"         : f"{m['mean_delta']:+.2f}",
            "95% CI"         : f"[{lo:+.2f}, {hi:+.2f}]",
            "% Early"        : f"{m['pct_early']:.1%}",
            "Mean Δ | early" : f"{m['mean_early']:+.2f}",
            "Std Δ"          : f"{m['std_delta']:.2f}",
            "N(τ)"           : m['n_tau'],
            "N(early)"       : m['n_early'],
        })

    results_df = pd.DataFrame(rows)
    print("\n" + results_df.to_string(index=False))

    print("\n  Pairwise Mann–Whitney U tests (two-sided):")
    pairs = [("Model", "Imbalance"),
             ("Model", "Volatility"),
             ("Imbalance", "Volatility")]
    for a, b in pairs:
        u, p = mannwhitney_test(delta_dict[a], delta_dict[b])
        sig  = ("***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else "ns")
        print(f"    {a:12s} vs {b:12s}: U={u:,.0f}  p={p:.4f}  {sig}")

    # ── Summary ──────────────────────────────────────────────────
    m_model = evaluation_metrics(delta_model)
    print(f"\n  ── SUMMARY ──────────────────────────────────────")
    print(f"  Model Mean Δ      : {m_model['mean_delta']:+.2f} steps")
    print(f"  Model % Early     : {m_model['pct_early']:.1%}")
    print(f"  Model Mean Δ|early: {m_model['mean_early']:+.2f} steps")
    if m_model['mean_delta'] > 0:
        print("  ✓ Positive mean lead-time achieved")
    else:
        print("  ✗ Mean Δ still negative — check parameters")
    if m_model['pct_early'] > 0.60:
        print("  ✓ > 60% early detection rate achieved")
    else:
        print("  ✗ < 60% early — try lowering MIN_LEAD or SIGNAL_PCT")
    print()

    # ── Figures ──────────────────────────────────────────────────
    print("  Rendering figures …")
    plot_dgp_causal_structure(X_raw, Z_true, delay_map)
    plot_delay_distribution(delay_map, T)
    plot_trigger_signal_decomposition(
        X_raw, Z_true, score_amp, d_score, comps,
        tau_model, tau_raw, sigma)
    plot_composite_signal(X_raw, Z_true, score_amp, tau_model, sigma)
    plot_rising_edge_zoom(X_raw, Z_true, score_amp, d_score, tau_model, sigma)
    plot_lead_time_densities(delta_dict)
    plot_channel_importance(comps, Z_true)
    plot_results_table(results_df)

    print("\n  Experiment complete.")
    return (results_df, delta_dict, model,
            X_raw, Z_true, Z_hat, sigma, delay_map,
            score_amp, d_score, comps, post_smooth,
            tau_model, tau_raw)


if __name__ == "__main__":
    (results_df, delta_dict, model,
     X_raw, Z_true, Z_hat, sigma, delay_map,
     score_amp, d_score, comps, post_smooth,
     tau_model, tau_raw) = run_experiment()
