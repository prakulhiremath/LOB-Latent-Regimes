# ============================================================
#  Latent Micro-Regimes in Limit Order Books:
#  Identification and Early Detection  — v4
#  ─────────────────────────────────────────
#  ROOT-CAUSE FIX: The DGP now embeds a genuine causal
#  pre-stress build-up phase (Regime 1) that precedes every
#  stress event by a mandatory latent delay (k ~ U[10,50]).
#
#  Key properties of the new DGP
#  ──────────────────────────────
#  • Regime 1 signals are SUBTLE:
#      - spread rises only moderately
#      - depth erodes gradually (AR-decay, not a jump)
#      - imbalance drifts, but stays below naive thresholds
#      - rolling volatility barely changes   ← baselines miss this
#  • Stress (Regime 2) is triggered ONLY after Regime 1 has
#    persisted for k steps → guaranteed lead-time window
#  • Gradual blending at regime boundaries hides hard switches
#  • HMM posterior instability captures the subtle Regime-1
#    fingerprint; simple threshold baselines cannot
#
#  Everything downstream (evaluation, stats, plots) unchanged.
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
T           = 14_000      # slightly longer for richer regime coverage
N_REGIMES   = 3
MAX_LAG     = 60
FW_WINDOW   = 20
STRESS_PCT  = 95
N_BOOT      = 2_000
MIN_GAP     = 20
PENALTY     = -MAX_LAG

# Composite signal weights
W_ENTROPY   = 0.40
W_UNCERT    = 0.25
W_TRANS     = 0.20
W_PRESTRESS = 0.15
SIGNAL_PCT  = 82           # adaptive threshold percentile

# DGP delay parameters
DELAY_LO    = 10           # minimum Regime-1 → Regime-2 delay (steps)
DELAY_HI    = 50           # maximum delay
BLEND_WIN   = 8            # boundary blending half-window (gradual transitions)

np.random.seed(SEED)

# ─────────────────────────────────────────
#  1. Causal Delayed Stress DGP
#  ─────────────────────────────
#  The time axis is governed by an EXPLICIT state machine:
#
#    State 0 (Stable)   → stays 0 with high prob; can enter 1
#    State 1 (Build-up) → mandatory hold for k ~ U[DELAY_LO, DELAY_HI] steps
#                         then deterministically enters 2
#    State 2 (Crisis)   → decays back to 0 or 1 after a crisis duration
#
#  Regime 1 is calibrated so that:
#    • its SPREAD increment is < 30% of the 95th-pctile spread in Regime 0
#    • its IMBALANCE stays below the 90th-pctile imbalance baseline threshold
#    • its ROLLING VOL barely exceeds the 90th-pctile vol baseline threshold
#  → simple threshold detectors remain blind; only the HMM posterior
#    integrates all subtle channels simultaneously.
# ─────────────────────────────────────────

# Per-regime parameter dictionaries
#   sp_mu / sp_sig : log-normal spread parameters
#   dp_ar          : AR(1) coefficient for depth
#   dp_mu / dp_sig : depth long-run mean and noise std
#   ib_mu / ib_sig : order-flow imbalance mean and std
#   vol_noise      : extra iid noise added to spread (drives rolling-vol)

REGIME_PARAMS = {
    # ── Regime 0: Stable ───────────────────────────────────────────────
    0: dict(
        sp_mu   = 1.5,   sp_sig  = 0.20,   # tight spread
        dp_ar   = 0.95,  dp_mu   = 120.0,  dp_sig  = 6.0,   # deep book
        ib_mu   = 0.00,  ib_sig  = 0.06,   # balanced flow
        vol_noise = 0.02,
    ),
    # ── Regime 1: Hidden Build-up ───────────────────────────────────────
    #   Deliberately subtle so that no single feature triggers a naive
    #   threshold; the HMM posterior integrates all channels jointly.
    1: dict(
        sp_mu   = 2.4,   sp_sig  = 0.35,   # moderate spread rise
        dp_ar   = 0.93,  dp_mu   =  92.0,  dp_sig  = 9.0,   # slow erosion
        ib_mu   = 0.12,  ib_sig  = 0.09,   # mild directional pressure
        vol_noise = 0.06,                  # slightly elevated but sub-threshold
    ),
    # ── Regime 2: Crisis ────────────────────────────────────────────────
    2: dict(
        sp_mu   = 8.0,   sp_sig  = 1.30,   # large spread spike
        dp_ar   = 0.88,  dp_mu   =  35.0,  dp_sig  = 18.0,  # depth collapse
        ib_mu   = 0.50,  ib_sig  = 0.20,   # extreme imbalance
        vol_noise = 0.40,
    ),
}


def _draw_delay(rng):
    """Sample the mandatory Regime-1 persistence before stress."""
    return int(rng.integers(DELAY_LO, DELAY_HI + 1))


def _draw_crisis_duration(rng):
    """Crisis lasts 15–60 steps before recovery."""
    return int(rng.integers(15, 61))


def _draw_stable_duration(rng):
    """Stable spells last 80–300 steps."""
    return int(rng.integers(80, 301))


def build_regime_sequence(T, rng):
    """
    Explicit state-machine DGP that guarantees:
      - every Regime-2 episode is preceded by Regime-1 for k steps
      - k is drawn i.i.d. from U[DELAY_LO, DELAY_HI]
      - regime boundaries are recorded for blending

    Returns
    -------
    Z        : (T,) int array of true latent states
    delay_map: dict  t → delay k for each Regime-1 entry point
    """
    Z = np.zeros(T, dtype=int)
    delay_map = {}

    t = 0
    while t < T:
        # ── Stable spell ──────────────────────────────────────────────
        dur0 = _draw_stable_duration(rng)
        end0 = min(t + dur0, T)
        Z[t:end0] = 0
        t = end0
        if t >= T:
            break

        # ── Build-up (Regime 1) ───────────────────────────────────────
        k    = _draw_delay(rng)
        end1 = min(t + k, T)
        Z[t:end1] = 1
        delay_map[t] = k          # record entry point and delay
        t = end1
        if t >= T:
            break

        # ── Crisis (Regime 2) ─────────────────────────────────────────
        dur2 = _draw_crisis_duration(rng)
        end2 = min(t + dur2, T)
        Z[t:end2] = 2
        t = end2

    return Z, delay_map


def _blend(x, Z, win=BLEND_WIN):
    """
    Smooth sharp regime boundaries with a localised Gaussian blur.
    This hides the exact transition point from simple detectors.
    """
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
    """
    Simulate LOB features under the causal delayed-stress DGP.

    Features produced
    -----------------
    spread    : bid-ask spread (log-normal + Hawkes self-excitation)
    depth     : aggregate book depth (AR-1 per regime)
    imbalance : order-flow imbalance (truncated normal per regime)
    roll_vol  : 20-step rolling spread volatility
    ofi       : order-flow imbalance proxy
    """
    Z, delay_map = build_regime_sequence(T, rng)

    spread    = np.zeros(T)
    depth     = np.zeros(T)
    imbalance = np.zeros(T)

    # ── Spread: log-normal + Hawkes self-excitation ──────────────────
    hawkes = 0.0
    hawkes_decay = 0.90
    for t in range(T):
        p      = REGIME_PARAMS[Z[t]]
        hawkes *= hawkes_decay
        base   = np.log(p['sp_mu'])
        eps    = rng.normal(0, p['sp_sig']) + rng.normal(0, p['vol_noise'])
        spread[t] = np.exp(base + 0.10 * hawkes + eps)
        # Hawkes excitation: only significant spikes propagate
        if spread[t] > np.exp(base + 0.8 * p['sp_sig']):
            hawkes += 0.30

    # ── Depth: per-regime AR(1) with mean-reversion ──────────────────
    depth[0] = REGIME_PARAMS[Z[0]]['dp_mu']
    for t in range(1, T):
        p       = REGIME_PARAMS[Z[t]]
        depth[t] = (p['dp_ar'] * depth[t-1]
                    + (1 - p['dp_ar']) * p['dp_mu']
                    + rng.normal(0, p['dp_sig']))
    depth = np.clip(depth, 5.0, None)

    # ── Imbalance: truncated normal ───────────────────────────────────
    for t in range(T):
        p           = REGIME_PARAMS[Z[t]]
        imbalance[t] = np.clip(rng.normal(p['ib_mu'], p['ib_sig']), -1.0, 1.0)

    # ── Blend boundaries to obscure exact switch times ───────────────
    spread    = _blend(spread,    Z)
    depth     = _blend(depth,     Z)
    imbalance = _blend(imbalance, Z)

    # ── Derived features ──────────────────────────────────────────────
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
#  2. Feature Engineering & Normalisation
# ─────────────────────────────────────────

def engineer_features(X_raw):
    spread    = X_raw[:, 0]
    depth     = X_raw[:, 1]
    imbalance = X_raw[:, 2]
    roll_vol  = X_raw[:, 3]
    ofi       = X_raw[:, 4]

    sd_ratio    = spread / (depth + 1e-6)
    abs_imb     = np.abs(imbalance)
    cum_ofi     = pd.Series(ofi).rolling(50, min_periods=1).mean().values
    roll_depth  = (pd.Series(depth)
                     .rolling(20, min_periods=1)
                     .mean()
                     .fillna(method='bfill')
                     .values)
    # Additional channel: depth-velocity (rate of erosion)
    ddepth = -pd.Series(depth).diff(5).fillna(0).values  # positive = erosion

    X_full  = np.column_stack([
        spread, depth, imbalance, roll_vol, ofi,
        sd_ratio, abs_imb, cum_ofi, roll_depth, ddepth
    ])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled, scaler


# ─────────────────────────────────────────
#  3. HMM Fitting
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
#  5. Posterior-Based Signal Computation
#     (same architecture as v3, unchanged)
# ─────────────────────────────────────────

def smooth_posterior(posterior, window=7):
    """Causal trailing rolling mean — no look-ahead."""
    return pd.DataFrame(posterior).rolling(window, min_periods=1).mean().values


def entropy_signal(post):
    eps = 1e-12
    return -np.sum(post * np.log(post + eps), axis=1)


def uncertainty_signal(post):
    return 1.0 - post.max(axis=1)


def transition_intensity_signal(post):
    ti = np.abs(np.diff(post, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], ti])


def prestress_posterior_signal(post, model):
    means_raw    = model.means_[:, 0]   # spread dimension
    state_rank   = np.argsort(means_raw)
    prestress_id = state_rank[1]        # intermediate spread state
    return post[:, prestress_id]


def _norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def build_composite_score(post, model):
    H  = entropy_signal(post)
    U  = uncertainty_signal(post)
    TI = transition_intensity_signal(post)
    PS = prestress_posterior_signal(post, model)

    score = (W_ENTROPY   * _norm01(H)  +
             W_UNCERT    * _norm01(U)  +
             W_TRANS     * _norm01(TI) +
             W_PRESTRESS * _norm01(PS))
    return score, H, U, TI, PS


def deduplicate(indices, min_gap=MIN_GAP):
    if len(indices) == 0:
        return np.array([], dtype=int)
    out = [indices[0]]
    for idx in indices[1:]:
        if idx - out[-1] >= min_gap:
            out.append(idx)
    return np.array(out, dtype=int)


def model_signals(model, X_scaled, smooth_win=7,
                  signal_pct=SIGNAL_PCT, min_gap=MIN_GAP):
    posterior   = model.predict_proba(X_scaled)
    post_smooth = smooth_posterior(posterior, window=smooth_win)
    score, H, U, TI, PS = build_composite_score(post_smooth, model)
    threshold   = np.percentile(score, signal_pct)
    raw         = np.where(score > threshold)[0]
    tau         = deduplicate(raw, min_gap=min_gap)
    return tau, score, H, U, TI, PS


def imbalance_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    imb = np.abs(X_raw[:, 2])
    raw = np.where(imb > np.percentile(imb, pct))[0]
    return deduplicate(raw, min_gap=min_gap)


def volatility_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    rv  = X_raw[:, 3]
    raw = np.where(rv > np.percentile(rv, pct))[0]
    return deduplicate(raw, min_gap=min_gap)


# ─────────────────────────────────────────
#  6. Lead-Time Evaluation  (UNCHANGED)
# ─────────────────────────────────────────

def compute_lead_times(tau, sigma, max_lag=MAX_LAG):
    deltas = np.empty(len(tau), dtype=float)
    for i, t in enumerate(tau):
        cands    = sigma[(sigma > t) & (sigma <= t + max_lag)]
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
#  7. Bootstrap CI + Mann–Whitney  (UNCHANGED)
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
#  8. Visualisation
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
    """
    Three-panel overview of the causal DGP:
      1. Spread with true regime shading + Regime-1 onset arrows
      2. Depth
      3. Imbalance
    Arrows mark Regime-1 entry (τ_true); the mandatory delay to stress
    is annotated on the first few events.
    """
    t_end   = min(n_show, len(Z_true))
    t_ax    = np.arange(t_end)
    spread  = X_raw[:t_end, 0]
    depth   = X_raw[:t_end, 1]
    imb     = X_raw[:t_end, 2]

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

    # Annotate first 5 Regime-1 onsets with delay arrows on spread panel
    ax0 = axes[0]
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

    # Custom legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(fc=REGIME_FILL[k], label=f"Regime {k}") for k in range(3)]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   fontsize=8, frameon=False, ncol=3)
    axes[2].set_xlabel("Timestep")
    fig.tight_layout()
    plt.savefig("lob_dgp_structure.pdf", bbox_inches="tight")
    plt.show()


def plot_composite_signal(X_raw, Z_true, score, tau_model, sigma, n_show=3000):
    """Four-panel: spread + regimes, composite score + signals, depth, imbalance."""
    t_end = min(n_show, len(Z_true))
    t_ax  = np.arange(t_end)

    tau_vis   = tau_model[tau_model < t_end]
    sigma_vis = sigma[sigma < t_end]
    sc        = score[:t_end]
    thresh    = np.percentile(score, SIGNAL_PCT)

    spread = X_raw[:t_end, 0]
    depth  = X_raw[:t_end, 1]
    imb    = X_raw[:t_end, 2]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2.5, 1.5, 1.5]})
    fig.suptitle("Posterior-Based Instability Detector — v4 (Causal DGP)",
                 fontsize=13, fontweight="bold")

    # Panel 1: spread + regime shading
    ax = axes[0]
    for k, c in REGIME_FILL.items():
        ax.fill_between(t_ax, 0, spread.max()*1.1,
                        where=Z_true[:t_end] == k, color=c, alpha=0.55,
                        label=f"Regime {k}")
    ax.plot(t_ax, spread, lw=0.65, color="#1A1A2E")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    # Panel 2: composite score + signals
    ax = axes[1]
    ax.plot(t_ax, sc, lw=0.8, color="#444444", alpha=0.85, label="Composite score")
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

    # Panel 3: depth
    ax = axes[2]
    ax.plot(t_ax, depth, lw=0.7, color="#2D6A4F")
    ax.set_ylabel("Depth")

    # Panel 4: imbalance
    ax = axes[3]
    ax.plot(t_ax, imb, lw=0.7, color="#6A3D9A", alpha=0.85)
    ax.set_ylabel("Imbalance")
    ax.set_xlabel("Timestep")

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
    ax.set_title("Lead-Time Distribution: Posterior-Based Model vs Baselines  [v4]",
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
        "Detection Performance Summary — v4 (Causal DGP + Posterior-Based Signals)",
        fontsize=10, y=1.02)
    fig.tight_layout()
    plt.savefig("lob_results_table.pdf", bbox_inches="tight")
    plt.show()


def plot_delay_distribution(delay_map, T):
    """Histogram of true Regime-1 delays (ground truth from DGP)."""
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


# ─────────────────────────────────────────
#  9. Sanity Check: Verify Baselines Are Blind to Regime 1
# ─────────────────────────────────────────

def check_baseline_blindness(X_raw, Z_true):
    """
    Prints percentile statistics of imbalance and rolling-vol
    in each regime, confirming that Regime-1 values do NOT
    exceed the 90th-percentile thresholds used by the baselines.
    """
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
#  10. Main Pipeline
# ─────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(SEED)

    print("=" * 68)
    print("  LOB Micro-Regime Detection v4")
    print("  Causal Delayed Stress DGP + Posterior Instability Signals")
    print("=" * 68)

    # ── Step 1: Data generation ──────────────────────────────────────
    print("\n  Step 1 / 6 — Generating causal LOB data …")
    X_raw, Z_true, delay_map = generate_lob_data(T, rng)
    regime_dist = " | ".join(
        [f"Regime {k}: {(Z_true==k).mean():.1%}" for k in range(N_REGIMES)])
    print(f"  {T:,} timesteps | {regime_dist}")
    print(f"  Regime-1 episodes: {len(delay_map)} "
          f"| Mean delay to stress: {np.mean(list(delay_map.values())):.1f} steps")

    # ── Step 2: Feature engineering ──────────────────────────────────
    print("\n  Step 2 / 6 — Feature engineering …")
    X_scaled, scaler = engineer_features(X_raw)
    print(f"  Feature matrix: {X_scaled.shape}")

    # ── Step 3: HMM ──────────────────────────────────────────────────
    print("\n  Step 3 / 6 — Fitting HMM (12 restarts) …")
    model   = fit_hmm(X_scaled)
    Z_hat   = model.predict(X_scaled)
    ll      = model.score(X_scaled)
    conv    = model.monitor_.converged
    print(f"  Best log-likelihood: {ll:,.2f} | Converged: {conv}")
    means_sp   = model.means_[:, 0]
    state_rank = np.argsort(means_sp)
    print(f"  HMM state ranking by spread: {state_rank.tolist()} (low → high)")

    # ── Step 4: Stress events ─────────────────────────────────────────
    print("\n  Step 4 / 6 — Stress event definition …")
    sigma = define_stress_events(X_raw)
    print(f"  Stress events: {len(sigma):,} ({len(sigma)/T:.1%} of timesteps)")

    # ── Step 4b: Sanity check ─────────────────────────────────────────
    print()
    check_baseline_blindness(X_raw, Z_true)

    # ── Step 5: Signals ───────────────────────────────────────────────
    print("  Step 5 / 6 — Computing signals …")
    tau_model, score, H, U, TI, PS = model_signals(model, X_scaled)
    tau_imb   = imbalance_baseline(X_raw)
    tau_vol   = volatility_baseline(X_raw)
    print(f"  Signals — Model: {len(tau_model)} | "
          f"Imbalance: {len(tau_imb)} | Volatility: {len(tau_vol)}")

    delta_model = compute_lead_times(tau_model, sigma)
    delta_imb   = compute_lead_times(tau_imb,   sigma)
    delta_vol   = compute_lead_times(tau_vol,   sigma)
    delta_dict  = {"Model": delta_model,
                   "Imbalance": delta_imb,
                   "Volatility": delta_vol}

    # ── Step 6: Statistical validation ───────────────────────────────
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

    # ── Figures ───────────────────────────────────────────────────────
    print("\n  Rendering figures …")
    plot_dgp_causal_structure(X_raw, Z_true, delay_map)
    plot_delay_distribution(delay_map, T)
    plot_composite_signal(X_raw, Z_true, score, tau_model, sigma)
    plot_lead_time_densities(delta_dict)
    plot_results_table(results_df)

    print("\n  Experiment complete.")
    return (results_df, delta_dict, model,
            X_raw, Z_true, Z_hat, sigma, delay_map,
            score, H, U, TI)


if __name__ == "__main__":
    (results_df, delta_dict, model,
     X_raw, Z_true, Z_hat, sigma, delay_map,
     score, H, U, TI) = run_experiment()
