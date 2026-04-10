# ============================================================
#  Latent Micro-Regimes in Limit Order Books:
#  Identification and Early Detection  — v3
#  ─────────────────────────────────────────
#  KEY UPGRADE over v2:
#    Hard regime-switch detection → Pre-transition instability
#    detection via HMM posterior-based composite signal.
#
#  Signal components (all from posterior probabilities):
#    H_t  = Shannon entropy of p(z | x_t)            [primary]
#    U_t  = 1 - max_z p(z | x_t)                     [uncertainty]
#    TI_t = ||p(z|x_t) - p(z|x_{t-1})||_1            [transition intensity]
#    PS_t = pre-stress state posterior (regime 1)     [regime-specific]
#
#  Composite score = weighted average → adaptive percentile threshold
#  + minimum-gap deduplication → sparse, meaningful signals τ
#
#  Evaluation pipeline (UNCHANGED from v2):
#    - leakage-free stress events
#    - lead-time Δ = σ − τ
#    - bootstrap CI + Mann–Whitney
#    - baselines (imbalance + volatility) with same dedup
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
T           = 12_000
N_REGIMES   = 3
MAX_LAG     = 60       # evaluation window (timesteps)
FW_WINDOW   = 20       # forward window for stress label
STRESS_PCT  = 95       # percentile for stress threshold
N_BOOT      = 2_000
MIN_GAP     = 20       # min timesteps between two signals (dedup)
PENALTY     = -MAX_LAG

# Composite signal weights
W_ENTROPY   = 0.40
W_UNCERT    = 0.25
W_TRANS     = 0.20
W_PRESTRESS = 0.15

# Detection threshold: fire when composite > this percentile
SIGNAL_PCT  = 82

np.random.seed(SEED)

# ─────────────────────────────────────────
#  1. Structured Latent Regime Generator
#     Regime 0 = Normal  (baseline, calm)
#     Regime 1 = Pre-stress build-up  ← harbinger state
#     Regime 2 = Crisis / stress peak
#
#  Causal chain:  0 → 1 → 2 → (0 or 1)
# ─────────────────────────────────────────

REGIME_PARAMS = {
    0: dict(sp_mu=1.5,  sp_sig=0.25, dp_mu=120, dp_sig=10,
            ib_mu=0.00, ib_sig=0.07, vb=0.35),
    1: dict(sp_mu=3.2,  sp_sig=0.55, dp_mu= 88, dp_sig=16,
            ib_mu=0.22, ib_sig=0.13, vb=1.10),
    2: dict(sp_mu=7.5,  sp_sig=1.20, dp_mu= 42, dp_sig=22,
            ib_mu=0.48, ib_sig=0.22, vb=2.90),
}


def build_causal_transition_matrix() -> np.ndarray:
    P = np.array([
        [0.970, 0.028, 0.002],
        [0.060, 0.900, 0.040],
        [0.100, 0.150, 0.750],
    ])
    return P / P.sum(axis=1, keepdims=True)


def simulate_latent_chain(T, P, pi0, rng):
    K = P.shape[0]
    Z = np.empty(T, dtype=int)
    Z[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        Z[t] = rng.choice(K, p=P[Z[t - 1]])
    return Z


def generate_lob_data(T, rng):
    P   = build_causal_transition_matrix()
    pi0 = np.array([0.85, 0.12, 0.03])
    Z   = simulate_latent_chain(T, P, pi0, rng)

    spread    = np.zeros(T)
    depth     = np.zeros(T)
    imbalance = np.zeros(T)

    hawkes = 0.0
    decay  = 0.88
    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        hawkes = hawkes * decay
        eps    = rng.normal(0, p['sp_sig'])
        spread[t] = np.exp(np.log(p['sp_mu']) + 0.12 * hawkes + eps)
        if spread[t] > np.exp(np.log(p['sp_mu']) + p['sp_sig']):
            hawkes += 0.25

    phi      = 0.93
    depth[0] = REGIME_PARAMS[Z[0]]['dp_mu']
    for t in range(1, T):
        p = REGIME_PARAMS[Z[t]]
        depth[t] = phi * depth[t-1] + (1-phi) * p['dp_mu'] + rng.normal(0, p['dp_sig'])
    depth = np.clip(depth, 5, None)

    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        imbalance[t] = np.clip(rng.normal(p['ib_mu'], p['ib_sig']), -1, 1)

    roll_vol = pd.Series(spread).pct_change().rolling(20).std().fillna(0).values
    ofi = imbalance * np.abs(np.diff(spread, prepend=spread[0]))

    X = np.column_stack([spread, depth, imbalance, roll_vol, ofi])
    return X, Z


# ─────────────────────────────────────────
#  2. Feature Engineering & Normalisation
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
    roll_depth = pd.Series(depth).rolling(20).mean().fillna(method='bfill').values

    X_full = np.column_stack([
        spread, depth, imbalance, roll_vol, ofi,
        sd_ratio, abs_imb, cum_ofi, roll_depth
    ])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled, scaler


# ─────────────────────────────────────────
#  3. HMM Fitting
# ─────────────────────────────────────────

def fit_hmm(X, n_components=N_REGIMES, n_restarts=10, rng_seed=SEED):
    best_score, best_model = -np.inf, None
    for k in range(n_restarts):
        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = "full",
            n_iter          = 300,
            tol             = 1e-6,
            random_state    = rng_seed + k,
            init_params     = "stmc",
            params          = "stmc",
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score, best_model = score, model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("HMM fitting failed.")
    return best_model


# ─────────────────────────────────────────
#  4. Stress Event Definition (no leakage)
# ─────────────────────────────────────────

def define_stress_events(X_raw, fw=FW_WINDOW, pct=STRESS_PCT):
    spread    = X_raw[:, 0]
    threshold = np.percentile(spread, pct)
    sigma = [t for t in range(len(spread) - fw)
             if np.mean(spread[t+1:t+fw+1]) > threshold]
    return np.array(sigma, dtype=int)


# ─────────────────────────────────────────
#  5. Posterior-Based Signal Computation
#     *** CORE UPGRADE ***
#
#  Four posterior-derived measures fused into a composite score.
#  Signals fire when the composite exceeds an adaptive threshold.
# ─────────────────────────────────────────

def smooth_posterior(posterior, window=5):
    """Causal rolling average to reduce HMM jitter (no look-ahead)."""
    return pd.DataFrame(posterior).rolling(window, min_periods=1).mean().values


def entropy_signal(post):
    """H_t = -∑ p_k log p_k  (high = uncertain = pre-transition)"""
    eps = 1e-12
    return -np.sum(post * np.log(post + eps), axis=1)


def uncertainty_signal(post):
    """U_t = 1 - max_k p_k  (high = diffuse posterior)"""
    return 1.0 - post.max(axis=1)


def transition_intensity_signal(post):
    """TI_t = L1 distance between consecutive posteriors."""
    diff = np.abs(np.diff(post, axis=0))
    ti   = diff.sum(axis=1)
    return np.concatenate([[0.0], ti])


def prestress_posterior_signal(post, model):
    """PS_t = posterior mass on the pre-stress (intermediate) HMM state."""
    means_raw    = model.means_[:, 0]        # spread dimension
    state_rank   = np.argsort(means_raw)
    prestress_id = state_rank[1]             # intermediate spread state
    return post[:, prestress_id]


def build_composite_score(post, model,
                           w_h=W_ENTROPY, w_u=W_UNCERT,
                           w_ti=W_TRANS,  w_ps=W_PRESTRESS):
    """
    Weighted composite of four posterior-derived instability signals.
    Each component is min-max normalised before weighting so that
    differences in scale do not bias the fusion.
    """
    H  = entropy_signal(post)
    U  = uncertainty_signal(post)
    TI = transition_intensity_signal(post)
    PS = prestress_posterior_signal(post, model)

    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    score = (w_h  * _norm(H)  +
             w_u  * _norm(U)  +
             w_ti * _norm(TI) +
             w_ps * _norm(PS))
    return score, H, U, TI, PS


def deduplicate(indices, min_gap=MIN_GAP):
    """Keep only the first index in each cluster within min_gap."""
    if len(indices) == 0:
        return indices
    out = [indices[0]]
    for idx in indices[1:]:
        if idx - out[-1] >= min_gap:
            out.append(idx)
    return np.array(out, dtype=int)


def model_signals(model, X_scaled,
                  smooth_win=5,
                  signal_pct=SIGNAL_PCT,
                  min_gap=MIN_GAP):
    """
    Generate early-warning signals τ from the composite instability score.

    Steps:
      1. Compute smoothed posterior (causal rolling mean)
      2. Build composite score from 4 posterior-derived measures
      3. Threshold at adaptive percentile (signal_pct)
      4. Deduplicate to enforce minimum gap

    Returns τ (signal times), composite score, and component signals.
    """
    posterior   = model.predict_proba(X_scaled)
    post_smooth = smooth_posterior(posterior, window=smooth_win)

    score, H, U, TI, PS = build_composite_score(post_smooth, model)

    threshold = np.percentile(score, signal_pct)
    raw       = np.where(score > threshold)[0]
    tau       = deduplicate(raw, min_gap=min_gap)

    return tau, score, H, U, TI, PS


def imbalance_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    imb       = np.abs(X_raw[:, 2])
    threshold = np.percentile(imb, pct)
    raw       = np.where(imb > threshold)[0]
    return deduplicate(raw, min_gap=min_gap)


def volatility_baseline(X_raw, pct=90, min_gap=MIN_GAP):
    roll_vol  = X_raw[:, 3]
    threshold = np.percentile(roll_vol, pct)
    raw       = np.where(roll_vol > threshold)[0]
    return deduplicate(raw, min_gap=min_gap)


# ─────────────────────────────────────────
#  6. Lead-Time Evaluation (UNCHANGED)
# ─────────────────────────────────────────

def compute_lead_times(tau, sigma, max_lag=MAX_LAG):
    deltas = np.empty(len(tau), dtype=float)
    for i, t in enumerate(tau):
        cands = sigma[(sigma > t) & (sigma <= t + max_lag)]
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
#  7. Bootstrap CI + Mann–Whitney (UNCHANGED)
# ─────────────────────────────────────────

def bootstrap_ci(deltas, stat_fn=np.mean, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    rng  = np.random.default_rng(seed)
    boot = np.array([stat_fn(rng.choice(deltas, size=len(deltas), replace=True))
                     for _ in range(n_boot)])
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


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


def plot_composite_signal(X_raw, Z_true, score, H, U, TI, tau_model,
                           sigma, n_show=3000):
    """
    Five-panel diagnostic:
      1. Spread + true regime shading
      2. Composite instability score + threshold + signal fires
      3. Entropy H_t
      4. Uncertainty U_t + Transition intensity TI_t
      5. Pre-stress posterior PS_t
    """
    t_end = min(n_show, len(Z_true))
    t_ax  = np.arange(t_end)
    spread  = X_raw[:t_end, 0]

    tau_vis   = tau_model[tau_model < t_end]
    sigma_vis = sigma[sigma < t_end]

    regime_colors = {0: "#DDEEFF", 1: "#FFF3CD", 2: "#FFDDDD"}

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2.5, 1.5, 1.5, 1.5]})
    fig.suptitle("Posterior-Based Pre-Transition Instability Detection\n"
                 "Latent Micro-Regimes in Limit Order Books (v3)",
                 fontsize=13, y=1.01, fontweight="bold")

    # ── Panel 1: Spread + true regime shading ──
    ax = axes[0]
    for k, c in regime_colors.items():
        ax.fill_between(t_ax, 0, spread.max()*1.1,
                        where=Z_true[:t_end] == k, color=c, alpha=0.55,
                        label=f"Regime {k}")
    ax.plot(t_ax, spread, lw=0.6, color="#1A1A2E")
    ax.set_ylabel("Bid-Ask Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    # ── Panel 2: Composite score + signals ──
    ax = axes[1]
    sc = score[:t_end]
    thresh = np.percentile(score, SIGNAL_PCT)
    ax.plot(t_ax, sc, lw=0.8, color="#444444", alpha=0.85, label="Composite score")
    ax.axhline(thresh, color="#FF8800", lw=1.2, ls="--",
               label=f"{SIGNAL_PCT}th pct threshold")
    ax.fill_between(t_ax, thresh, sc, where=sc > thresh,
                    color=PALETTE["Model"], alpha=0.18)
    ax.vlines(tau_vis, sc.min(), sc.max(),
              color=PALETTE["Model"], lw=1.0, alpha=0.7, label="Signal τ (model)")
    ax.vlines(sigma_vis, sc.min(), sc.max(),
              color=PALETTE["Imbalance"], lw=0.6, ls=":", alpha=0.4, label="Stress event σ")
    ax.set_ylabel("Instability Score")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # ── Panel 3: Entropy ──
    ax = axes[2]
    ax.plot(t_ax, H[:t_end], lw=0.7, color="#6A3D9A", alpha=0.85)
    ax.set_ylabel("Entropy H_t")

    # ── Panel 4: Uncertainty + Transition Intensity ──
    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(t_ax, U[:t_end],  lw=0.7, color="#1F78B4", alpha=0.9, label="Uncertainty U_t")
    ax2.plot(t_ax, TI[:t_end], lw=0.7, color="#33A02C", alpha=0.6, label="Trans. Intensity TI_t")
    ax.set_ylabel("Uncertainty U_t", color="#1F78B4")
    ax2.set_ylabel("TI_t", color="#33A02C")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False, loc="upper right")

    # ── Panel 5: Pre-stress posterior ──
    ax = axes[4]
    # Recompute PS for display
    posterior   = None  # computed via score pipeline; approximate via reading from score
    ax.set_ylabel("(see note)")
    ax.set_xlabel("Timestep")
    # Annotate instead
    ax.text(0.5, 0.5,
            "PS_t (pre-stress posterior) is fused into composite score above.\n"
            "See build_composite_score() for component breakdown.",
            ha="center", va="center", transform=ax.transAxes, fontsize=9,
            color="#555555", style="italic")
    ax.set_yticks([])

    fig.tight_layout()
    plt.savefig("lob_diagnostic.pdf", bbox_inches="tight")
    plt.show()


def plot_lead_time_densities(delta_dict, max_lag=MAX_LAG):
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid = np.linspace(-max_lag - 5, max_lag + 5, 800)

    for i, (name, deltas) in enumerate(delta_dict.items()):
        color = PALETTE[name]
        valid = deltas[deltas > PENALTY]
        if len(valid) > 5:
            kde = gaussian_kde(valid, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), lw=2.4, color=color, label=name)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.14, color=color)
        pf = np.mean(deltas <= PENALTY)
        ax.annotate(f"{name}  missed={pf:.1%}",
                    xy=(-max_lag + 1, 0.006 * (i + 1)),
                    color=color, fontsize=9, fontweight="bold")

    ax.axvline(0, color="gray", lw=1.2, ls="--", label="Zero lead-time")
    ax.set_xlabel("Lead time Δ (timesteps before stress event)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Lead-Time Distribution: Posterior-Based Model vs Baselines",
                 fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(-max_lag - 2, max_lag + 2)
    fig.tight_layout()
    plt.savefig("lob_lead_time.pdf", bbox_inches="tight")
    plt.show()


def plot_results_table(results_df):
    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.7)
    for j in range(len(results_df.columns)):
        tbl[0, j].set_facecolor("#2C6FAC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight model row
    for j in range(len(results_df.columns)):
        tbl[1, j].set_facecolor("#EDF4FF")
    fig.suptitle("Detection Performance Summary (v3 — Posterior-Based Signals)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    plt.savefig("lob_results_table.pdf", bbox_inches="tight")
    plt.show()


def plot_component_contributions(score, H, U, TI, n_show=3000):
    """Show how each posterior component contributes to the composite score."""
    t_end = min(n_show, len(score))
    t_ax  = np.arange(t_end)

    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
    fig.suptitle("Posterior Signal Components vs Composite Instability Score",
                 fontsize=12, fontweight="bold")

    items = [
        (axes[0, 0], _norm(H[:t_end]),  "Entropy H_t (normalised)",      "#6A3D9A"),
        (axes[0, 1], _norm(U[:t_end]),  "Uncertainty U_t (normalised)",   "#1F78B4"),
        (axes[1, 0], _norm(TI[:t_end]), "Trans. Intensity TI_t (norm.)",  "#33A02C"),
        (axes[1, 1], score[:t_end],     "Composite Score (weighted sum)",  "#2C6FAC"),
    ]
    thresh = np.percentile(score, SIGNAL_PCT)
    for ax, y, title, color in items:
        ax.plot(t_ax, y, lw=0.7, color=color, alpha=0.85)
        if "Composite" in title:
            ax.axhline(thresh, color="#FF8800", lw=1.1, ls="--",
                       label=f"Threshold ({SIGNAL_PCT}th pct)")
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Value")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 1].set_xlabel("Timestep")
    fig.tight_layout()
    plt.savefig("lob_components.pdf", bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────
#  9. Main Pipeline
# ─────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(SEED)

    print("=" * 66)
    print("  LOB Micro-Regime Detection v3 — Posterior-Based Early Warning")
    print("=" * 66)

    # ── Step 1: Data generation ──
    print("\n  Step 1 / 6 — Generating causal LOB data …")
    X_raw, Z_true = generate_lob_data(T, rng)
    dist = " | ".join([f"State {k}: {(Z_true==k).mean():.1%}" for k in range(N_REGIMES)])
    print(f"  {T:,} timesteps | {dist}")

    # ── Step 2: Features ──
    print("\n  Step 2 / 6 — Feature engineering …")
    X_scaled, scaler = engineer_features(X_raw)
    print(f"  Feature matrix: {X_scaled.shape}")

    # ── Step 3: HMM ──
    print("\n  Step 3 / 6 — Fitting HMM (10 restarts) …")
    model  = fit_hmm(X_scaled)
    Z_hat  = model.predict(X_scaled)
    ll     = model.score(X_scaled)
    print(f"  Best log-likelihood: {ll:,.2f} | Converged: {model.monitor_.converged}")
    means_sp   = model.means_[:, 0]
    state_rank = np.argsort(means_sp)
    print(f"  HMM state ranking by spread: {state_rank.tolist()} (low → high)")

    # ── Step 4: Stress events ──
    print("\n  Step 4 / 6 — Stress event definition …")
    sigma = define_stress_events(X_raw)
    print(f"  Stress events: {len(sigma):,} ({len(sigma)/T:.1%} of timesteps)")

    # ── Step 5: Signal computation ──
    print("\n  Step 5 / 6 — Computing posterior-based instability signals …")
    tau_model, score, H, U, TI, PS = model_signals(model, X_scaled)
    tau_imb   = imbalance_baseline(X_raw)
    tau_vol   = volatility_baseline(X_raw)

    print(f"  Signals — Model: {len(tau_model)} | "
          f"Imbalance: {len(tau_imb)} | Volatility: {len(tau_vol)}")

    delta_model = compute_lead_times(tau_model, sigma)
    delta_imb   = compute_lead_times(tau_imb,   sigma)
    delta_vol   = compute_lead_times(tau_vol,   sigma)

    delta_dict = {"Model": delta_model, "Imbalance": delta_imb, "Volatility": delta_vol}

    # ── Step 6: Statistical validation ──
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
    pairs = [("Model", "Imbalance"), ("Model", "Volatility"), ("Imbalance", "Volatility")]
    for a, b in pairs:
        u, p = mannwhitney_test(delta_dict[a], delta_dict[b])
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {a:12s} vs {b:12s}: U={u:,.0f}  p={p:.4f}  {sig}")

    # ── Visualisation ──
    print("\n  Rendering figures …")
    plot_composite_signal(X_raw, Z_true, score, H, U, TI, tau_model, sigma)
    plot_component_contributions(score, H, U, TI)
    plot_lead_time_densities(delta_dict)
    plot_results_table(results_df)

    print("\n  Experiment complete.")
    return results_df, delta_dict, model, X_raw, Z_true, Z_hat, sigma, score, H, U, TI


if __name__ == "__main__":
    (results_df, delta_dict, model,
     X_raw, Z_true, Z_hat, sigma,
     score, H, U, TI) = run_experiment()
