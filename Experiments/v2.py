# ============================================================
#  Latent Micro-Regimes in Limit Order Books:
#  Identification and Early Detection  — v2
#  ─────────────────────────────────────────
#  Key changes over v1:
#    1. Regime 1 = pre-stress build-up, Regime 2 = crisis
#       → HMM entry into regime 1 is the early-warning signal
#    2. Detection via posterior ENTROPY rise, not hard-switch
#    3. Minimum-gap deduplication (no signal flooding)
#    4. Posterior probability smoothing for cleaner signals
#    5. Baselines use identical deduplication
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
MIN_GAP     = 15       # minimum timesteps between two signals (dedup)
PENALTY     = -MAX_LAG

np.random.seed(SEED)

# ─────────────────────────────────────────
#  1. Structured Latent Regime Generator
#     Regime 0 = Normal  (baseline, calm)
#     Regime 1 = Pre-stress build-up  ← the key early-warning state
#     Regime 2 = Crisis / stress peak
#
#  Causal chain:  0 → 1 → 2 → (0 or 1)
#  The HMM must learn that regime 1 is a harbinger of regime 2.
#  We encode this by:
#    • Making spread/depth in regime 1 intermediate
#    • Making transitions 0→1 more likely than 0→2 directly
#    • Stress events defined on FUTURE spread (no leakage)
# ─────────────────────────────────────────

REGIME_PARAMS = {
    # spread:  mu, sigma (log-normal)
    # depth:   mu, sigma (AR-1)
    # imb:     mu, sigma (truncated normal)
    # vol_base: scaling factor
    0: dict(sp_mu=1.5,  sp_sig=0.25, dp_mu=120, dp_sig=10, ib_mu=0.00, ib_sig=0.07, vb=0.35),
    1: dict(sp_mu=3.2,  sp_sig=0.55, dp_mu= 88, dp_sig=16, ib_mu=0.22, ib_sig=0.13, vb=1.10),
    2: dict(sp_mu=7.5,  sp_sig=1.20, dp_mu= 42, dp_sig=22, ib_mu=0.48, ib_sig=0.22, vb=2.90),
}

def build_causal_transition_matrix() -> np.ndarray:
    """
    Transition matrix with causal structure:
      - 0 → 1 much more likely than 0 → 2   (stress builds gradually)
      - 1 → 2 is the crisis trigger
      - 2 → 0 or 1 (recovery)
    """
    P = np.array([
        [0.970, 0.028, 0.002],   # Normal   → mostly stays
        [0.060, 0.900, 0.040],   # Pre-stress → can escalate
        [0.100, 0.150, 0.750],   # Crisis   → decays back
    ])
    # Normalise rows (safety)
    return P / P.sum(axis=1, keepdims=True)


def simulate_latent_chain(T, P, pi0, rng):
    K = P.shape[0]
    Z = np.empty(T, dtype=int)
    Z[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        Z[t] = rng.choice(K, p=P[Z[t - 1]])
    return Z


def generate_lob_data(T, rng):
    """
    Simulate LOB features under a causal Markov-switching model.

    Hawkes-like self-excitation on spread ensures bursts cluster
    without trivially revealing regime via a single feature.
    """
    P   = build_causal_transition_matrix()
    pi0 = np.array([0.85, 0.12, 0.03])
    Z   = simulate_latent_chain(T, P, pi0, rng)

    spread    = np.zeros(T)
    depth     = np.zeros(T)
    imbalance = np.zeros(T)

    # Spread: log-normal + Hawkes excitation
    hawkes = 0.0
    decay  = 0.88
    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        hawkes = hawkes * decay
        eps    = rng.normal(0, p['sp_sig'])
        spread[t] = np.exp(np.log(p['sp_mu']) + 0.12 * hawkes + eps)
        if spread[t] > np.exp(np.log(p['sp_mu']) + p['sp_sig']):
            hawkes += 0.25

    # Depth: mean-reverting AR(1)
    phi      = 0.93
    depth[0] = REGIME_PARAMS[Z[0]]['dp_mu']
    for t in range(1, T):
        p = REGIME_PARAMS[Z[t]]
        depth[t] = phi * depth[t-1] + (1-phi) * p['dp_mu'] + rng.normal(0, p['dp_sig'])
    depth = np.clip(depth, 5, None)

    # Imbalance: truncated normal
    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        imbalance[t] = np.clip(rng.normal(p['ib_mu'], p['ib_sig']), -1, 1)

    # Rolling spread vol (20-step)
    roll_vol = pd.Series(spread).pct_change().rolling(20).std().fillna(0).values

    # OFI proxy
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
#  5. Signal Computation (the core upgrade)
# ─────────────────────────────────────────

def state_entropy(posterior):
    """Shannon entropy of the posterior state distribution at each step."""
    eps = 1e-12
    return -np.sum(posterior * np.log(posterior + eps), axis=1)


def deduplicate(indices, min_gap=MIN_GAP):
    """
    Keep only the FIRST index in each cluster of indices
    that are within min_gap of each other.
    This prevents signal flooding.
    """
    if len(indices) == 0:
        return indices
    out  = [indices[0]]
    for idx in indices[1:]:
        if idx - out[-1] >= min_gap:
            out.append(idx)
    return np.array(out, dtype=int)


def model_signals(model, X_scaled, Z_hat,
                  entropy_pct=80, smooth_window=5, min_gap=MIN_GAP):
    """
    Uncertainty-based early-warning signal.

    Strategy (three complementary triggers, unioned then deduped):
      A) Entropy spike: posterior uncertainty rises sharply
         → captures "the HMM is unsure", a known pre-transition marker
      B) Pre-stress state entry: Z_hat transitions INTO the intermediate
         state (state with 2nd-highest mean spread)
         → aligns detection with causal regime structure
      C) Entropy trend: rolling entropy slope turns positive
         → catches gradual build-up before a hard switch

    All triggers are deduplicated so N(τ) stays controlled.
    """
    posterior = model.predict_proba(X_scaled)          # (T, K)

    # ── Smooth posterior to reduce HMM jitter ──
    smooth_post = pd.DataFrame(posterior).rolling(
        smooth_window, min_periods=1, center=True).mean().values

    entropy = state_entropy(smooth_post)

    # ── Identify the "pre-stress" HMM state ──
    # We rank states by their mean spread (feature 0 in raw space)
    means_raw = model.means_[:, 0]                    # spread dim of HMM means
    state_rank = np.argsort(means_raw)                 # [low, mid, high]
    prestress_state = state_rank[1]                    # intermediate spread state

    # ── Trigger A: entropy spike ──
    ent_thresh = np.percentile(entropy, entropy_pct)
    sig_A = np.where(entropy > ent_thresh)[0]

    # ── Trigger B: entry into pre-stress state ──
    enters_prestress = (Z_hat[1:] == prestress_state) & (Z_hat[:-1] != prestress_state)
    sig_B = np.where(enters_prestress)[0]

    # ── Trigger C: entropy slope turns positive ──
    ent_slope = pd.Series(entropy).diff(5).fillna(0).values
    slope_thresh = np.percentile(ent_slope[ent_slope > 0], 70)
    sig_C = np.where(ent_slope > slope_thresh)[0]

    # ── Union + deduplicate ──
    all_signals = np.unique(np.concatenate([sig_A, sig_B, sig_C]))
    tau = deduplicate(all_signals, min_gap=min_gap)
    return tau


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
#  6. Lead-Time Evaluation
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
        mean_delta  = float(np.mean(deltas)),
        pct_early   = float(np.mean(valid)),
        mean_early  = float(np.mean(deltas[valid])) if valid.any() else 0.0,
        std_delta   = float(np.std(deltas)),
        n_tau       = int(len(deltas)),
        n_early     = int(valid.sum()),
    )


# ─────────────────────────────────────────
#  7. Bootstrap CI + Mann–Whitney
# ─────────────────────────────────────────

def bootstrap_ci(deltas, stat_fn=np.mean, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    boot = np.array([stat_fn(rng.choice(deltas, size=len(deltas), replace=True))
                     for _ in range(n_boot)])
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def mannwhitney_test(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# ─────────────────────────────────────────
#  8. Publication-Quality Visualisation
# ─────────────────────────────────────────

PALETTE = {"Model": "#2C6FAC", "Imbalance": "#D94F3D", "Volatility": "#5AAE61"}

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "figure.dpi"       : 150,
})


def plot_entropy_and_regimes(X_raw, Z_true, Z_hat, entropy, tau_model,
                              sigma, n_show=2500):
    """
    Four-panel diagnostic:
      1. Spread with true regime shading
      2. Posterior entropy (with signal thresholds and triggers marked)
      3. Inferred HMM state
      4. Depth with stress events
    """
    t_end = min(n_show, len(Z_true))
    t_ax  = np.arange(t_end)
    spread = X_raw[:t_end, 0]
    depth  = X_raw[:t_end, 1]

    tau_vis   = tau_model[tau_model < t_end]
    sigma_vis = sigma[sigma < t_end]

    regime_colors = {0: "#DDEEFF", 1: "#FFF3CD", 2: "#FFDDDD"}

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1, 2]})

    # Panel 1 — Spread + true shading
    ax = axes[0]
    for k, c in regime_colors.items():
        ax.fill_between(t_ax, 0, spread.max()*1.1,
                        where=Z_true[:t_end]==k, color=c, alpha=0.55,
                        label=f"State {k}")
    ax.plot(t_ax, spread, lw=0.7, color="#1A1A2E")
    ax.set_ylabel("Bid-Ask Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    ax.set_title("True Regimes · Posterior Entropy · Inferred States · Market Depth",
                 fontsize=12, pad=8)

    # Panel 2 — Entropy + model triggers
    ax = axes[1]
    ent_vis = entropy[:t_end]
    ax.plot(t_ax, ent_vis, lw=0.9, color="#555555", alpha=0.85, label="Entropy")
    ent_thresh = np.percentile(entropy, 80)
    ax.axhline(ent_thresh, color="#FF8800", lw=1.0, ls="--", label="80th pct threshold")
    ax.vlines(tau_vis, ent_vis.min(), ent_vis.max(),
              color=PALETTE["Model"], lw=0.7, alpha=0.6, label="Model signal τ")
    ax.set_ylabel("Posterior Entropy")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel 3 — Inferred HMM state
    ax = axes[2]
    ax.step(t_ax, Z_hat[:t_end], lw=0.9, color=PALETTE["Model"])
    ax.set_ylabel("HMM State")
    ax.set_yticks([0, 1, 2])

    # Panel 4 — Depth + stress events
    ax = axes[3]
    ax.plot(t_ax, depth, lw=0.7, color="#2D6A4F")
    ax.vlines(sigma_vis, depth.min(), depth.max(),
              color=PALETTE["Imbalance"], lw=0.7, alpha=0.5, label="Stress event σ")
    ax.set_ylabel("Market Depth")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    plt.show()


def plot_lead_time_densities(delta_dict, max_lag=MAX_LAG):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_grid = np.linspace(-max_lag-5, max_lag+5, 600)

    for i, (name, deltas) in enumerate(delta_dict.items()):
        color  = PALETTE[name]
        valid  = deltas[deltas > PENALTY]
        if len(valid) > 5:
            kde = gaussian_kde(valid, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), lw=2.2, color=color, label=name)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.13, color=color)
        pf = np.mean(deltas <= PENALTY)
        ax.annotate(f"missed={pf:.1%}", xy=(-max_lag+1, 0.004*(i+1)),
                    color=color, fontsize=8.5)

    ax.axvline(0, color="gray", lw=1.0, ls="--", label="Zero lead-time")
    ax.set_xlabel("Lead time Δ (timesteps)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Lead-Time Distribution: Model vs Baselines", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(-max_lag-2, max_lag+2)
    fig.tight_layout()
    plt.show()


def plot_results_table(results_df):
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.6)
    for j in range(len(results_df.columns)):
        tbl[0, j].set_facecolor("#2C6FAC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────
#  9. Main Pipeline
# ─────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(SEED)

    print("=" * 62)
    print("  Step 1 / 6 — Generating causal LOB data …")
    X_raw, Z_true = generate_lob_data(T, rng)
    dist = " | ".join([f"State {k}: {(Z_true==k).mean():.1%}" for k in range(N_REGIMES)])
    print(f"  {T:,} timesteps | {dist}")

    print("\n  Step 2 / 6 — Feature engineering …")
    X_scaled, scaler = engineer_features(X_raw)
    print(f"  Feature matrix: {X_scaled.shape}")

    print("\n  Step 3 / 6 — Fitting HMM (10 restarts) …")
    model  = fit_hmm(X_scaled)
    Z_hat  = model.predict(X_scaled)
    ll     = model.score(X_scaled)
    print(f"  Best log-likelihood: {ll:,.2f} | Converged: {model.monitor_.converged}")

    # Identify which HMM state corresponds to each true regime
    # (sorted by mean spread value in the HMM)
    means_sp = model.means_[:, 0]
    state_rank = np.argsort(means_sp)
    print(f"  HMM state ranking by spread: {state_rank.tolist()} "
          f"(low→high spread)")

    print("\n  Step 4 / 6 — Stress event definition …")
    sigma = define_stress_events(X_raw)
    print(f"  Stress events: {len(sigma):,} ({len(sigma)/T:.1%} of timesteps)")

    print("\n  Step 5 / 6 — Computing signals …")
    # Model signals: uncertainty + pre-stress entry
    posterior = model.predict_proba(X_scaled)
    smooth_post = pd.DataFrame(posterior).rolling(5, min_periods=1, center=True).mean().values
    entropy  = state_entropy(smooth_post)

    tau_model = model_signals(model, X_scaled, Z_hat)
    tau_imb   = imbalance_baseline(X_raw)
    tau_vol   = volatility_baseline(X_raw)

    print(f"  Signals — Model: {len(tau_model)} | "
          f"Imbalance: {len(tau_imb)} | Volatility: {len(tau_vol)}")

    delta_model = compute_lead_times(tau_model, sigma)
    delta_imb   = compute_lead_times(tau_imb,   sigma)
    delta_vol   = compute_lead_times(tau_vol,   sigma)

    delta_dict = {"Model": delta_model, "Imbalance": delta_imb, "Volatility": delta_vol}

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
    pairs = [("Model","Imbalance"),("Model","Volatility"),("Imbalance","Volatility")]
    for a, b in pairs:
        u, p = mannwhitney_test(delta_dict[a], delta_dict[b])
        sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        print(f"    {a:12s} vs {b:12s}: U={u:,.0f}  p={p:.4f}  {sig}")

    print("\n  Rendering figures …")
    plot_entropy_and_regimes(X_raw, Z_true, Z_hat, entropy, tau_model, sigma)
    plot_lead_time_densities(delta_dict)
    plot_results_table(results_df)

    print("\n  Experiment complete.")
    return results_df, delta_dict, model, X_raw, Z_true, Z_hat, sigma, entropy


if __name__ == "__main__":
    (results_df, delta_dict, model,
     X_raw, Z_true, Z_hat, sigma, entropy) = run_experiment()
