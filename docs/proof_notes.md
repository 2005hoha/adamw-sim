# AdamW Proof Notes

Proof notes corresponding to Section 4 of the paper.

## Main Theorem Skeleton (Zhou et al. 2024)

**Assumptions:**

1. **L-smoothness:** $f$ is $L$-smooth: $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$.

2. **Bounded gradients:** $\|\nabla f(\theta_t)\|_\infty \le G_\infty$ for all $t$.

3. **Weight decay:** $\lambda > 0$ is fixed.

**Key Lemma (Descent Lemma):**

By $L$-smoothness:
$$
f(\theta_{t+1}) \le f(\theta_t) + \langle \nabla f(\theta_t), \theta_{t+1} - \theta_t \rangle + \frac{L}{2}\|\theta_{t+1} - \theta_t\|^2
$$

**Moment Bound:**

From the definition of $v_t$ and the $\epsilon$ lower bound:
$$
\frac{1}{\sqrt{\hat{v}_t} + \epsilon} \le \frac{1}{\epsilon},\quad \|\hat{m}_t\| \le c_\infty
$$

**Bias Correction:**
The bias correction terms $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ approach 1 as $t \to \infty$.

**Decoupled Decay:**
The AdamW update can be decomposed as "gradient step + contraction term",
where the contraction term $- \eta \lambda \theta_t$ can be absorbed into a dynamic regularized loss.

**Telescoping:**
Summing over $t = 1, \dots, T$:
$$
\frac{1}{T} \sum_{t=1}^T \mathbb{E}\|\nabla f(\theta_t)\|^2 \le \frac{C}{\sqrt{T}} + \frac{C'}{T}
$$

**Step Size Selection:**
- Constant step size: $O(\epsilon^{-4})$
- Decaying step size: $O(\epsilon^{-4} \log(1/\epsilon))$

## Connections to Other Results

| Reference | Key Contribution |
|---|---|
| Reddi et al. 2018 | Adam divergence counterexample; AMSGrad fix |
| Chen et al. 2019 | $O(\log T/\sqrt{T})$ for Adam-type under mild conditions |
| Zou et al. 2019 | Verifiable convergence conditions for Adam/RMSProp |
| Zhou et al. 2024 (TPAMI) | AdamW convergence + dynamic regularization |
| Li et al. 2025 (2505.11840) | $l_1$ rate $O(d/K^{1/4})$ for AdamW |
| Arjevani et al. 2019 | Nonconvex lower bound $\Omega(\epsilon^{-4})$ |