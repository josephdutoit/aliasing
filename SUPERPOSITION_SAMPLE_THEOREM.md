# A sample-dependent preference reversal for superposition

This is a proved benchmark proposition for fixed encoders and a specified head
rule. It is not a theorem about which encoder gradient descent learns, nor a
novelty claim. It gives a concrete result to generalize to continuous pairs.

## Setup

Let s1,s2 be independent Bernoulli(p), with 0<p<=1/2, and f=s1+s2.
Use one hidden dimension in both competing representations:

- Sharing: z=s1-s2, readout phi=(1,ReLU(z),ReLU(-z)).
- Dropping: z=s1, readout phi=(1,s1); the identically zero ReLU channel is removed.

For every n, fit the minimum Euclidean-norm empirical least-squares head from
zero. This is the same rule as experiment 06, extended to incompatible samples
by least squares. All losses below are unnormalized squared population risk.

Write v=p(1-p) and k=1-2v. The model-insufficiency terms are

\[
B_{\rm share}=\frac{4v^2}{k},\qquad B_{\rm drop}=v.
\]

Thus sharing has lower B exactly when

\[
p<p_B=\frac{1-1/\sqrt3}{2}\approx0.2113248654
\]

in the sparse regime. With one training sample, interpolation is always feasible
and the original frozen-feature GAD is exact. Moreover

\[
\mathbb E A_{\rm share}=4v^2,
\qquad \mathbb E A_{\rm drop}=vk.
\]

The B and A comparisons have the same boundary p_B.

## Proposition: limited data can reverse the population preference

Define

\[
p_1=\frac{19-\sqrt{337}}4\approx0.1606100623.
\]

For every p in the nonempty interval (p_1,p_B):

1. Sharing has strictly smaller B and strictly smaller expected A at n=1.
2. Its actual minimum-norm head nevertheless has strictly larger expected
   population risk at n=1. The difference is explained by D.
3. There is an integer N(p) such that sharing has strictly smaller expected
   population risk for every n>=N(p), under the same least-squares fitting rule.

The assertion is an eventual reversal. It does not assert a unique crossing or
monotonicity of either learning curve.

### Proof at n=1

Let rho=(1-p)^2, t=p(1-p), and q=p^2. Enumerating the four possible training
examples gives

\[
\begin{aligned}
\mathbb E R_{\rm share}(1)
&=\tfrac52\rho t+8\rho q+\tfrac12t^2+\tfrac{13}2tq,\\
\mathbb E R_{\rm drop}(1)
&=\tfrac{13}4\rho t+5\rho q+\tfrac14t^2+3tq.
\end{aligned}
\]

Subtracting and using independence,

\[
\boxed{
\mathbb E R_{\rm share}(1)-\mathbb E R_{\rm drop}(1)
=\frac{p(1-p)}4(-3+19p-2p^2).
}
\]

This is positive for p>p_1 on (0,1/2]. Combining it with the B/A inequalities
for p<p_B proves the first two claims. Since R=B+A+D holds at n=1, the D
increase exceeds the combined B/A reductions.

### Proof of eventual reversal

Both readout classes can fit arbitrary values on their distinct encoded cells:
three cells for sharing and two for dropping. Every cell has positive mass in
the stated interval. Eventually all cells are observed, and the fitted function
on each cell is its empirical mean target. These converge to their conditional
population means.

Before full coverage, minimum-norm extrapolation to the finitely many unseen
cells is bounded, since targets lie in [0,2] and there are finitely many possible
observed-cell sets. Dominated convergence therefore gives

\[
\mathbb E R_{\rm share}(n)\to B_{\rm share},
\qquad
\mathbb E R_{\rm drop}(n)\to B_{\rm drop}.
\]

Their limiting difference is strictly negative for p<p_B. Hence it is negative
for every sufficiently large integer n. This proves claim 3.

## Exact risk formula for arbitrary sample sizes

This formula applies to any finite encoded-cell distribution whose distinct
readout rows are linearly independent. Let cell c have mass pi_c, mean mu_c,
conditional target variance sigma_c^2, and readout row V_c. Let N_c be its
training count and O={c:N_c>0}. Define

\[
T_O=V V_O^\dagger.
\]

Conditional on the counts, the mean head predictions are T_O mu_O. Empirical
means in different cells have independent conditional variances sigma_c^2/N_c.
Therefore

\[
\boxed{
\mathbb E[R\mid(N_c)]
=B+
\sum_c\pi_c\big((T_O\mu_O)_c-\mu_c\big)^2
+\sum_{j\in O}\frac{\sigma_j^2}{N_j}
\sum_c\pi_c(T_O)_{cj}^2.
}
\]

Taking the finite multinomial expectation gives exact learning curves.
Experiment 08 evaluates these expectations using occupancy inclusion-exclusion,
which avoids enumerating all training sequences. Its sampling excess is

\[
V_n=\sum_c\pi_c\sigma_c^2\,
\mathbb E\!\left[\frac{\mathbf1_{N_c>0}}{N_c}\right].
\]

The remaining excess U_n is prediction error on unobserved cells, including
variance transferred from observed cell means. Thus E[R]=B+V_n+U_n. At n=1,
V_1=E[A] and U_1=E[D]. On noninterpolating datasets, V_n,U_n are projected-label
least-squares quantities, not the original label-interpolation A,D.

For the sum task the cell parameters are:

| Encoder | Cell masses | Conditional means | Conditional variances |
|---|---|---|---|
| Sharing | (rho+q,t,t) | (2q/(rho+q),1,1) | (4rho q/(rho+q)^2,0,0) |
| Dropping | (1-p,p) | (t/(1-p),(t+2q)/p) | (rho t/(1-p)^2,tq/p^2) |

## An explicit sufficient sample bound

For sharing, put mu0=2q/k and

\[
c_0=\max\{(1/2-\mu_0)^2,(2/3-\mu_0)^2\}.
\]

The ambiguous cell has N0~Binomial(n,k), so V_n=B_share E[1_{N0>0}/N0].
Using 1/j<=2/(j+1) for j>=1 gives

\[
V_n\le\frac{2B_{\rm share}}{(n+1)k}.
\]

An unseen singleton has population mean 1. Its fitted prediction is either an
empirical ambiguous-cell mean in [0,2], or 1/2, so its squared discrepancy is at
most 1. If the ambiguous cell is unseen, its fitted prediction is 1/2 or 2/3.
Hence

\[
U_n\le 2t(1-t)^n+k c_0(1-k)^n.
\]

It follows that sharing is strictly better whenever

\[
\boxed{
\frac{2B_{\rm share}}{(n+1)k}
+2t(1-t)^n+k c_0(1-k)^n
<B_{\rm drop}-B_{\rm share}.
}
\]

The left side decreases with n and tends to zero, providing an explicit
sufficient N(p) throughout the theorem interval.

At p=1/5, B_share=64/425 and B_drop=4/25. The expected one-sample risks are
322/625 and 304/625 respectively. Here k=17/25, t=4/25, mu0=2/17 and
c0=(28/51)^2. Substituting n=47 into the boxed inequality proves it strictly
(the test checks the inequality with exact rational arithmetic). Thus sharing
is better for **every n>=47**. This is a sufficient bound, not the first crossing.

## Interpolation scope

Sharing collapses 00 and 11, whose sum targets differ. Their simultaneous
presence makes interpolation impossible, with feasibility probability

\[
(1-\rho)^n+(1-q)^n-(1-\rho-q)^n.
\]

The proof must therefore use ordinary least squares for the larger-n comparison.
It would be incorrect to claim an unconditional original-GAD decomposition at
all sample sizes. The numerical driver records this probability and leaves
original A,D undefined beyond its guaranteed n=1 calibration.

## Experiments and remaining theorem target

```bash
bash run.sh 08
```

This writes exact curves, sample-bound certificates and a figure in
`outputs/superposition_sample_boundary/`. Experiments 06 and 07 provide the
matched-budget intervention and continuous/multi-pair checks respectively.
The next substantive target is a nonasymptotic continuous-pair result in a
specified interpolation regime, explaining the boundary through residual
transfer and coverage rather than transplanting the binary threshold.

Finite-data superposition and memorization have prior literature, including
Henighan et al., *Superposition, Memorization, and Double Descent* (2023):
https://transformer-circuits.pub/2023/toy-double-descent/index.html
The proposition here is a controlled fixed-encoder comparison. Establishing
novelty requires a closer comparison and stronger continuous/multi-pair results.
