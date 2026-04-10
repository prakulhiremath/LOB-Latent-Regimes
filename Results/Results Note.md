## Results Note

The reported results are generated using a controlled synthetic limit order book (LOB) environment with a causal regime structure.

Key points:
- The data includes a latent build-up regime preceding observable stress with a delayed transition.
- Evaluation is performed using lead-time (Δ = σ − τ), measuring how early a signal occurs relative to stress.
- Precision corresponds to the fraction of detections that occur before stress (early detection rate).
- Coverage measures the fraction of stress events that are successfully detected early.

The proposed method prioritizes **early and reliable detection** over exhaustive coverage. As a result:
- Lead-times are strictly positive.
- Precision is high (often 100%).
- Coverage is moderate, reflecting a deliberate trade-off between early detection and signal frequency.

Baseline methods (imbalance and volatility) achieve higher coverage but produce negative lead-times, indicating reactive behavior.

All results are reproducible via the provided pipeline and reflect the designed causal structure of the environment.
