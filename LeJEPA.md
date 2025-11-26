# **LeJEPA**

## **1. The Core Idea**

LeJEPA is a **latent forward-dynamics learner** based on the JEPA philosophy:

> *Instead of predicting raw future states, predict how the future latent representation relates to the present latent representation.*

To support this, we created two encoders:

### **Context Encoder**

* Learns from masked observations
* Receives gradients
* Learns to extract *predictively useful* information
* Must infer the future even when part of the input is missing
* This forces invariance and generalization

### **Target Encoder (EMA Encoder)**

* Same architecture, but **updates slowly**
* Never receives gradients directly
* Stabilizes learning (“slow-moving teacher”)
* Prevents collapse
* Ensures the target embedding zₜ₊ₖ is a **stable training signal**

These two encoders are structurally identical but serve different *functional* purposes.

---

# **2. The Problems**

### **Problem 1 — Predicting absolute future states collapses**

If the model tries to directly predict:

```
z_hat → z_future
```

It collapses because:

* It has no reason to model structure
* Many future states look similar
* Prediction variance is low → encoder collapses to trivial features

### **Solution**

Predict **residual change**:

```
z_hat = z_t + Δz_pred
```

This matches biological intuition (neurons predict changes) and real-world physics (states evolve smoothly).

---

### **Problem 2 — Model learns shortcuts instead of dynamics**

Predictors will cheat by copying shallow cues from input → no real reasoning.

### **Solution: Feature masking**

We built a masking system:

* Mask coordinates (x, y)
* Mask velocities (vx, vy)
* Mask arbitrary 2-dims
* Mask single dims
* Mask all dims

This forces the model to infer missing information from dynamics patterns, not raw features.

---

### **Problem 3 — Stability during training**

The predictor learned too fast and destabilized the encoder.

### **Solution: EMA Target Encoder**

Use:

```
target = 0.995 * target + 0.005 * context
```

* The target encoder changes slowly
* It offers a *consistent target latent*
* Prevents collapse
* Improves long-term accuracy

This is the same trick used in BYOL, MAE, and LeCun’s JEPA.

---

### **Problem 4 — Predictive models lose long-term structure**

Predicting only Δz is not enough. Over multiple steps:

* Predictions drift
* Latents lose physical meaning
* PCA plots degrade

### **Solution: Dynamics Prior + Residual Predictor**

Added a tiny MLP that predicts a **default Δz** based only on the current latent:

```
Δz_prior = dynamics_prior(z_t)
Δz_residual = predictor(...)
Δz_total = Δz_prior + Δz_residual

z_hat = z_t + Δz_total
```

Benefits:

* Smoothness
* Physics bias
* Prevents drift
* Predictor only learns *corrections*, not full transitions

### **Problem 5 — Only training with fixed k is unrealistic**

Predicting only zₜ₊₁ is easy. Real systems have variable horizons.

### **Solution: Multi-k training**

Each iteration uses a different offset:

```
k ∈ {1, 2, 4, 8, 10}
```

Randomly sampled.

Predictor is conditioned on time-gap k via an embedding:

```
time_embed = Linear(1 → 4)
latent_delta = f(z, time_embed)
```
