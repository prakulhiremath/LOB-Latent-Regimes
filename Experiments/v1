# ============================================================
#  Latent Micro-Regimes in Limit Order Books:
#  Identification and Early Detection
#  ─────────────────────────────────────────
#  Research-grade pipeline — Colab-ready single script
#  Authors: [redacted for blind review]
# ============================================================
# Install (uncomment in Colab):
# !pip install hmmlearn scikit-learn scipy numpy pandas matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from hmmlearn.hmm import GaussianHMM

# ─────────────────────────────────────────
#  0. Global Configuration
# ─────────────────────────────────────────
SEED       = 42
T          = 12_000          # total timesteps
N_REGIMES  = 3               # latent states: Normal, Stressed, Crisis
MAX_LAG    = 60              # evaluation window (timesteps)
FW_WINDOW  = 20              # forward window for stress event definition
N_BOOT     = 2_000           # bootstrap replicates
STRESS_PCT = 95              # percentile threshold for stress definition

np.random.seed(SEED)

# ─────────────────────────────────────────
#  1. Structured Latent Regime Data Generator
# ─────────────────────────────────────────

def build_transition_matrix(persist: list[float]) -> np.ndarray:
    """
    Build a row-stochastic transition matrix from persistence probabilities.
    Off-diagonal mass is split evenly among other states.

    Parameters
    ----------
    persist : list of length N_REGIMES
        Self-transition probability for each state.

    Returns
    -------
    P : (N_REGIMES, N_REGIMES) ndarray
    """
    K = len(persist)
    P = np.zeros((K, K))
    for i, p in enumerate(persist):
        P[i, i] = p
        off = (1 - p) / (K - 1)
        for j in range(K):
            if j != i:
                P[i, j] = off
    return P


def simulate_latent_chain(T: int, P: np.ndarray, pi0: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a discrete-time Markov chain.

    Parameters
    ----------
    T   : int — number of timesteps
    P   : (K, K) transition matrix
    pi0 : (K,) initial distribution
    rng : numpy Generator

    Returns
    -------
    Z : (T,) int array of latent states
    """
    K = P.shape[0]
    Z = np.empty(T, dtype=int)
    Z[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        Z[t] = rng.choice(K, p=P[Z[t - 1]])
    return Z


# Regime-specific parameter sets
#   State 0 — Normal:   tight spread, deep book, balanced flow
#   State 1 — Stressed: wider spread, shallower book, directional flow
#   State 2 — Crisis:   very wide spread, thin book, extreme imbalance
REGIME_PARAMS = {
    #           spread_mu, spread_sig, depth_mu, depth_sig, imb_mu, imb_sig, vol_base
    0: dict(sp_mu=1.5,  sp_sig=0.30, dp_mu=120, dp_sig=12, ib_mu=0.00, ib_sig=0.08, vb=0.40),
    1: dict(sp_mu=3.5,  sp_sig=0.60, dp_mu= 85, dp_sig=18, ib_mu=0.18, ib_sig=0.14, vb=1.20),
    2: dict(sp_mu=7.0,  sp_sig=1.20, dp_mu= 45, dp_sig=22, ib_mu=0.40, ib_sig=0.22, vb=2.80),
}


def generate_lob_data(T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic LOB microstructure data driven by a latent Markov chain.

    Design choices:
    - Regime-conditional mean and volatility for each feature.
    - Spread follows a regime-switching log-normal process (always positive).
    - Depth follows a regime-switching AR(1) process (mean-reverting).
    - Order-flow imbalance is beta-distributed (bounded in [-1, 1]).
    - Cross-sectional noise is injected so no feature alone trivially reveals
      the regime.
    - A Hawkes-inspired self-exciting component adds burst clustering to spread.

    Returns
    -------
    X : (T, 5) feature matrix  [spread, depth, imbalance, roll_vol, ofi]
    Z : (T,)   true latent state sequence
    """
    # 1) Latent chain
    persist = [0.97, 0.93, 0.89]   # high persistence → realistic regimes
    P  = build_transition_matrix(persist)
    pi0 = np.array([0.80, 0.15, 0.05])
    Z  = simulate_latent_chain(T, P, pi0, rng)

    spread     = np.zeros(T)
    depth      = np.zeros(T)
    imbalance  = np.zeros(T)

    # ── Spread: log-normal with Hawkes-like self-excitation ──
    hawkes_intensity = np.zeros(T)
    decay = 0.90
    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        hawkes_intensity[t] = (hawkes_intensity[t - 1] * decay if t > 0 else 0)
        noise = rng.normal(0, p['sp_sig'])
        log_sp = np.log(p['sp_mu']) + 0.15 * hawkes_intensity[t] + noise
        spread[t] = np.exp(log_sp)
        if spread[t] > np.exp(np.log(p['sp_mu']) + p['sp_sig']):
            hawkes_intensity[t] += 0.30           # self-excite on spike

    # ── Depth: mean-reverting AR(1) per regime ──
    depth[0] = REGIME_PARAMS[Z[0]]['dp_mu']
    phi = 0.92                                    # AR coefficient
    for t in range(1, T):
        p = REGIME_PARAMS[Z[t]]
        depth[t] = phi * depth[t - 1] + (1 - phi) * p['dp_mu'] + rng.normal(0, p['dp_sig'])
    depth = np.clip(depth, 5, None)               # depth always positive

    # ── Order-flow imbalance: truncated normal ──
    for t in range(T):
        p = REGIME_PARAMS[Z[t]]
        raw = rng.normal(p['ib_mu'], p['ib_sig'])
        imbalance[t] = np.clip(raw, -1, 1)

    # ── Engineered features ──
    # Rolling 20-step realised volatility of spread
    roll_vol = pd.Series(spread).pct_change().rolling(20).std().fillna(0).values

    # Order-flow imbalance sign × magnitude (OFI proxy)
    ofi = imbalance * np.abs(np.diff(spread, prepend=spread[0]))

    X = np.column_stack([spread, depth, imbalance, roll_vol, ofi])
    return X, Z


# ─────────────────────────────────────────
#  2. Feature Engineering & Normalisation
# ─────────────────────────────────────────

def engineer_features(X_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Augment and normalise the raw feature matrix.

    Raw columns : [spread, depth, imbalance, roll_vol, ofi]
    Added       : [spread/depth ratio, |imbalance|, cumulative ofi (50-step)]

    Returns
    -------
    X_scaled : (T, n_features) normalised array
    scaler   : fitted StandardScaler (for reproducibility)
    """
    spread    = X_raw[:, 0]
    depth     = X_raw[:, 1]
    imbalance = X_raw[:, 2]
    roll_vol  = X_raw[:, 3]
    ofi       = X_raw[:, 4]

    spread_depth_ratio = spread / (depth + 1e-6)
    abs_imb            = np.abs(imbalance)
    cum_ofi            = pd.Series(ofi).rolling(50, min_periods=1).mean().values

    X_full = np.column_stack([
        spread, depth, imbalance, roll_vol, ofi,
        spread_depth_ratio, abs_imb, cum_ofi
    ])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    return X_scaled, scaler


# ─────────────────────────────────────────
#  3. HMM Fitting with Robust Initialisation
# ─────────────────────────────────────────

def fit_hmm(X: np.ndarray, n_components: int = N_REGIMES,
            n_restarts: int = 10, rng_seed: int = SEED) -> GaussianHMM:
    """
    Fit a Gaussian HMM with multiple random restarts and select
    the run with highest log-likelihood.

    Parameters
    ----------
    X           : normalised feature matrix
    n_components: number of latent states
    n_restarts  : number of random restarts
    rng_seed    : base seed

    Returns
    -------
    best_model : GaussianHMM with highest converged log-likelihood
    """
    best_score = -np.inf
    best_model = None

    for k in range(n_restarts):
        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = "full",
            n_iter          = 200,
            tol             = 1e-5,
            random_state    = rng_seed + k,
            init_params     = "stmc",
            params          = "stmc",
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM fitting failed across all restarts.")

    return best_model


# ─────────────────────────────────────────
#  4. Liquidity Stress Event Definition
# ─────────────────────────────────────────

def define_stress_events(X_raw: np.ndarray, fw: int = FW_WINDOW,
                          pct: float = STRESS_PCT) -> np.ndarray:
    """
    Identify liquidity stress events WITHOUT lookahead leakage.

    A timestep t is a stress event if the mean spread over
    [t+1, t+fw] exceeds the global spread 'pct'-percentile.

    The threshold is computed on the FULL spread series but
    the forward window ensures the label at t uses only future data.

    Parameters
    ----------
    X_raw : raw feature matrix (column 0 = spread)
    fw    : forward window length
    pct   : percentile for threshold

    Returns
    -------
    sigma : 1-D array of stress event timestep indices
    """
    spread    = X_raw[:, 0]
    threshold = np.percentile(spread, pct)
    sigma     = []
    for t in range(len(spread) - fw):
        if np.mean(spread[t + 1: t + fw + 1]) > threshold:
            sigma.append(t)
    return np.array(sigma, dtype=int)


# ─────────────────────────────────────────
#  5. Regime Transition Detection
# ─────────────────────────────────────────

def detect_transitions(Z_hat: np.ndarray) -> np.ndarray:
    """
    Return indices just *before* each detected regime change.

    Parameters
    ----------
    Z_hat : (T,) array of inferred (or true) state labels

    Returns
    -------
    tau : 1-D int array of transition indices
    """
    return np.where(np.diff(Z_hat) != 0)[0]


# ─────────────────────────────────────────
#  6. Lead-Time Evaluation
# ─────────────────────────────────────────

PENALTY = -MAX_LAG   # assigned delta when no stress event found in window


def compute_lead_times(tau: np.ndarray, sigma: np.ndarray,
                        max_lag: int = MAX_LAG) -> np.ndarray:
    """
    For each detected transition τ, find the first stress event σ
    in (τ, τ + max_lag].

    Returns
    -------
    deltas : (len(tau),) array
        Positive  → transition preceded stress (early detection)
        PENALTY   → no stress event in window (missed / false alarm)
    """
    deltas = np.empty(len(tau), dtype=float)
    for i, t in enumerate(tau):
        candidates = sigma[(sigma > t) & (sigma <= t + max_lag)]
        deltas[i] = (candidates[0] - t) if len(candidates) > 0 else PENALTY
    return deltas


def evaluation_metrics(deltas: np.ndarray, max_lag: int = MAX_LAG
                        ) -> dict:
    """
    Summarise lead-time performance.

    Metrics
    -------
    mean_delta   : mean lead time (penalised)
    pct_early    : fraction of transitions that preceded a stress event
    mean_early   : mean lead time conditional on early detection
    std_delta    : standard deviation of delta
    n_tau        : total number of detected transitions
    n_early      : number of early detections
    """
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
#  7. Baseline Detectors
# ─────────────────────────────────────────

def imbalance_baseline(X_raw: np.ndarray, pct: float = 90) -> np.ndarray:
    """
    Trigger when |order-flow imbalance| exceeds a percentile threshold.
    Noise is NOT added here; the baseline uses the same raw features as the
    HMM to ensure a fair comparison.
    """
    imb       = np.abs(X_raw[:, 2])
    threshold = np.percentile(imb, pct)
    return np.where(imb > threshold)[0]


def volatility_baseline(X_raw: np.ndarray, pct: float = 90) -> np.ndarray:
    """
    Trigger when rolling spread volatility exceeds a percentile threshold.
    """
    roll_vol  = X_raw[:, 3]
    threshold = np.percentile(roll_vol, pct)
    return np.where(roll_vol > threshold)[0]


# ─────────────────────────────────────────
#  8. Bootstrap Confidence Intervals
# ─────────────────────────────────────────

def bootstrap_ci(deltas: np.ndarray, stat_fn=np.mean,
                  n_boot: int = N_BOOT, alpha: float = 0.05,
                  seed: int = SEED) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for a scalar statistic.

    Returns
    -------
    (lower, upper) CI at (1-alpha) level
    """
    rng      = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(rng.choice(deltas, size=len(deltas), replace=True))
        for _ in range(n_boot)
    ])
    return (float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))))


def mannwhitney_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Two-sided Mann–Whitney U test (non-parametric, appropriate for
    skewed lead-time distributions).

    Returns (U-statistic, p-value).
    """
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# ─────────────────────────────────────────
#  9. Publication-Quality Visualisation
# ─────────────────────────────────────────

PALETTE = {
    "Model"      : "#2C6FAC",
    "Imbalance"  : "#D94F3D",
    "Volatility" : "#5AAE61",
}

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi"       : 150,
})


def plot_lead_time_densities(delta_dict: dict, max_lag: int = MAX_LAG,
                              save_path: str = None):
    """
    Overlapping KDE plots of lead-time distributions for each detector.

    Parameters
    ----------
    delta_dict : {'Model': deltas, 'Imbalance': deltas, 'Volatility': deltas}
    max_lag    : used to shade the penalty region
    save_path  : if provided, saves the figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x_grid = np.linspace(-max_lag - 5, max_lag + 5, 500)

    for name, deltas in delta_dict.items():
        color = PALETTE[name]
        # KDE on non-penalty values only (for readability)
        valid  = deltas[deltas > PENALTY]
        if len(valid) > 5:
            kde = gaussian_kde(valid, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), lw=2.2, color=color, label=name)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.12, color=color)

        # Mark penalty mass as a tick at the left edge
        penalty_frac = np.mean(deltas <= PENALTY)
        ax.annotate(
            f"  missed={penalty_frac:.1%}",
            xy=(-max_lag, 0),
            xytext=(-max_lag + 2, 0.012 * (list(delta_dict).index(name) + 1)),
            color=color, fontsize=8.5, va="center"
        )

    ax.axvline(0, color="gray", lw=1.0, ls="--", label="Zero lead-time")
    ax.set_xlabel("Lead time Δ (timesteps)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Lead-Time Distribution of Regime Detectors", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(-max_lag - 2, max_lag + 2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_regime_overlay(X_raw: np.ndarray, Z_true: np.ndarray,
                         Z_hat: np.ndarray, sigma: np.ndarray,
                         n_show: int = 3000, save_path: str = None):
    """
    Three-panel figure:
      Top    — bid-ask spread with true regime shading
      Middle — inferred HMM state sequence
      Bottom — depth time-series with stress events marked
    """
    t_end = min(n_show, len(Z_true))
    t_ax  = np.arange(t_end)
    spread = X_raw[:t_end, 0]
    depth  = X_raw[:t_end, 1]
    sigma_visible = sigma[sigma < t_end]

    regime_colors = {0: "#DDEEFF", 1: "#FFEEDD", 2: "#FFDDDD"}

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 2]})

    # ── Panel 1: Spread + true regime shading ──
    ax = axes[0]
    for k, color in regime_colors.items():
        mask = Z_true[:t_end] == k
        ax.fill_between(t_ax, 0, spread.max() * 1.1,
                         where=mask, color=color, alpha=0.5,
                         label=f"True state {k}")
    ax.plot(t_ax, spread, lw=0.7, color="#1A1A2E")
    ax.set_ylabel("Bid-Ask Spread")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    # ── Panel 2: Inferred HMM states ──
    ax = axes[1]
    ax.step(t_ax, Z_hat[:t_end], lw=0.9, color=PALETTE["Model"])
    ax.set_ylabel("HMM State")
    ax.set_yticks([0, 1, 2])

    # ── Panel 3: Depth + stress events ──
    ax = axes[2]
    ax.plot(t_ax, depth, lw=0.7, color="#2D6A4F")
    ax.vlines(sigma_visible, depth.min(), depth.max(),
              color=PALETTE["Imbalance"], lw=0.6, alpha=0.5, label="Stress event σ")
    ax.set_ylabel("Market Depth")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Simulated LOB: True Regimes, Inferred States & Stress Events",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_results_table(results_df: pd.DataFrame):
    """Render results DataFrame as a styled matplotlib table."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    col_labels = results_df.columns.tolist()
    rows       = results_df.values.tolist()
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.5)
    # Highlight header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C6FAC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────
#  10. Main Experimental Pipeline
# ─────────────────────────────────────────

def run_experiment():
    rng = np.random.default_rng(SEED)

    # ── 10.1 Data generation ──
    print("=" * 60)
    print("  Step 1 / 6 — Generating structured LOB data …")
    X_raw, Z_true = generate_lob_data(T, rng)
    print(f"  Generated {T:,} timesteps | "
          f"Regime distribution: "
          + " | ".join([f"State {k}: {(Z_true==k).mean():.1%}"
                        for k in range(N_REGIMES)]))

    # ── 10.2 Feature engineering ──
    print("\n  Step 2 / 6 — Engineering and normalising features …")
    X_scaled, scaler = engineer_features(X_raw)
    print(f"  Feature matrix shape: {X_scaled.shape}")

    # ── 10.3 HMM fitting ──
    print("\n  Step 3 / 6 — Fitting Gaussian HMM (multiple restarts) …")
    model = fit_hmm(X_scaled)
    Z_hat = model.predict(X_scaled)
    ll    = model.score(X_scaled)
    print(f"  Best log-likelihood: {ll:,.2f} | "
          f"Converged: {model.monitor_.converged}")

    # ── 10.4 Stress event definition ──
    print("\n  Step 4 / 6 — Defining leakage-free stress events …")
    sigma = define_stress_events(X_raw)
    print(f"  Stress events identified: {len(sigma):,} "
          f"({len(sigma)/T:.1%} of timesteps)")

    # ── 10.5 Transition detection for all detectors ──
    print("\n  Step 5 / 6 — Evaluating detectors …")

    tau_model = detect_transitions(Z_hat)
    tau_imb   = imbalance_baseline(X_raw)
    tau_vol   = volatility_baseline(X_raw)

    delta_model = compute_lead_times(tau_model, sigma)
    delta_imb   = compute_lead_times(tau_imb,   sigma)
    delta_vol   = compute_lead_times(tau_vol,   sigma)

    delta_dict = {
        "Model"     : delta_model,
        "Imbalance" : delta_imb,
        "Volatility": delta_vol,
    }

    # ── 10.6 Statistical validation ──
    print("\n  Step 6 / 6 — Statistical validation (bootstrap + MWU) …")

    rows = []
    for name, deltas in delta_dict.items():
        m  = evaluation_metrics(deltas)
        lo, hi = bootstrap_ci(deltas, stat_fn=np.mean)
        rows.append({
            "Detector"       : name,
            "Mean Δ"         : f"{m['mean_delta']:+.2f}",
            "95 % CI"        : f"[{lo:+.2f}, {hi:+.2f}]",
            "% Early"        : f"{m['pct_early']:.1%}",
            "Mean Δ | early" : f"{m['mean_early']:+.2f}",
            "Std Δ"          : f"{m['std_delta']:.2f}",
            "N(τ)"           : m['n_tau'],
            "N(early)"       : m['n_early'],
        })

    results_df = pd.DataFrame(rows)
    print("\n" + results_df.to_string(index=False))

    # Pairwise Mann–Whitney tests
    print("\n  Pairwise Mann–Whitney U tests (two-sided):")
    pairs = [("Model", "Imbalance"), ("Model", "Volatility"),
             ("Imbalance", "Volatility")]
    for a_name, b_name in pairs:
        u, p = mannwhitney_test(delta_dict[a_name], delta_dict[b_name])
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {a_name:12s} vs {b_name:12s}: U={u:,.0f}  p={p:.4f}  {sig}")

    # ── 10.7 Plots ──
    print("\n  Rendering publication-quality figures …")
    plot_regime_overlay(X_raw, Z_true, Z_hat, sigma)
    plot_lead_time_densities(delta_dict)
    plot_results_table(results_df)

    print("\n  Experiment complete.")
    return results_df, delta_dict, model, X_raw, Z_true, Z_hat, sigma


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    results_df, delta_dict, model, X_raw, Z_true, Z_hat, sigma = run_experiment()
