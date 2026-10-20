# Mathematical results for the aliasing experiments

This note separates exact finite-sample identities from statistical claims
about averages over generated problems. It also gives a controlled example
where feature superposition and representation aliasing can be distinguished.

## 1. Fixed-feature prediction-error identity

Let the training and test feature matrices be partitioned into modeled and
omitted columns:

\[
Z_{\rm tr}=[Z_m\;Z_u],\qquad Z_{\rm te}=[Z_{{\rm te},m}\;Z_{{\rm te},u}].
\]

For a fixed coefficient vector \(\theta=(\theta_m,\theta_u)\), suppose
training labels are

\[
y=Z_m\theta_m+Z_u\theta_u+\varepsilon,
\]

and fit the minimum-Euclidean-norm readout
\(\widehat\theta_m=Z_m^\dagger y\). The target on the test design is the
noiseless conditional mean
\(f_{\rm te}=Z_{{\rm te},m}\theta_m+Z_{{\rm te},u}\theta_u\). Then

\[
\begin{aligned}
\widehat f_{\rm te}-f_{\rm te}
= {}& Z_{{\rm te},m}(Z_m^\dagger Z_m-I)\theta_m \\
&+ Z_{{\rm te},m}Z_m^\dagger Z_u\theta_u \\
&- Z_{{\rm te},u}\theta_u \\
&+ Z_{{\rm te},m}Z_m^\dagger\varepsilon.
\end{aligned}
\]

The terms are called **data insufficiency**, **aliasing**, **model
insufficiency**, and **fitted noise**, respectively. This is an algebraic
identity for every fixed pair of train and test designs. It requires no
Gaussian, random-design, or full-rank assumption. The Moore-Penrose identity
\(Z_m^\dagger Z_m=P_{\mathrm{row}(Z_m)}\) shows that data insufficiency is
the test-visible part of the modeled coefficient vector in the training
nullspace.

### Proof

Substitute \(y\) into the fitted readout and subtract the target:

\[
Z_{{\rm te},m}Z_m^\dagger
(Z_m\theta_m+Z_u\theta_u+\varepsilon)
-Z_{{\rm te},m}\theta_m-Z_{{\rm te},u}\theta_u.
\]

Distributing \(Z_m^\dagger\) and grouping the four resulting terms gives the
displayed identity. The code computes the residual
\(\max_i|e_i-\sum_j e_{j,i}|\) as `reconstruction_error`; values near machine
precision check the implementation of this identity.

For component vectors \(e_1,\ldots,e_4\) over \(n_{\rm te}\) test examples,

\[
\frac{1}{n_{\rm te}}\left\|\sum_j e_j\right\|_2^2
=\sum_j\frac{\|e_j\|_2^2}{n_{\rm te}}
+\sum_{j<k}\frac{2\langle e_j,e_k\rangle}{n_{\rm te}}.
\]

Thus the four component energies are not generally additive. Their signed
pairwise cross terms are reported separately; the aggregate cross term is
their sum. A negative cross term represents cancellation and must not be
silently allocated among positive component energies.

The operator

\[
A_{\rm alias}=Z_{{\rm te},m}Z_m^\dagger Z_u
\]

is label-independent, and the realized aliasing vector is
\(A_{\rm alias}\theta_u\). The RMS-scaled operator norm
\(\|A_{\rm alias}\|_2/\sqrt{n_{\rm te}}\) is a worst-case gain per unit
omitted coefficient norm; it is not the realized aliasing risk. The latter is
\(\|A_{\rm alias}\theta_u\|_2^2/n_{\rm te}\).

The same result applies to a nonlinear fixed feature map
\(z=\phi(x)\) followed by a linear readout: use the feature matrices
\(Z_{\rm tr}=\phi(X_{\rm tr})\) and \(Z_{\rm te}=\phi(X_{\rm te})\).

## 2. Frozen-representation MLP null-space lemma

At a fixed checkpoint, let \(Z_{\rm tr}\in\mathbb R^{n\times h}\) and
\(Z_{\rm te}\in\mathbb R^{m\times h}\) be the hidden feature matrices,
and let \(a\in\mathbb R^h\) be the scalar-output head. Define

\[
P=Z_{\rm tr}^\dagger Z_{\rm tr},\qquad Q=I-P.
\]

Then \(P\) and \(Q\) are the orthogonal projectors onto
\(\mathrm{row}(Z_{\rm tr})\) and \(\ker(Z_{\rm tr})\), respectively, and

\[
Z_{\rm tr}Q=0,\qquad Z_{\rm tr}a=Z_{\rm tr}Pa.
\]

The two readouts \(a\) and \(Pa\) therefore agree exactly on the training
set, while their test predictions differ by

\[
Z_{\rm te}a-Z_{\rm te}Pa=Z_{\rm te}Qa.
\]

This proves that `test_null_response_rms` is the held-out RMS effect of the
current head's train-null component, conditional on the current hidden map.
It is not the total test risk and does not capture every way a learned
representation can alias inputs. The operator norm of \(Z_{\rm te}Q\)
measures whether any train-invisible head direction can affect the test set;
the vector \(Z_{\rm te}Qa\) measures the effect for the head actually
learned. RMS-scaled operator norms divide by \(\sqrt{n_{\rm te}}\).

The gap \(\|a-Z_{\rm tr}^\dagger y\|\) equals the null-head norm only when
the current model interpolates. Otherwise it also contains a row-space fit
residual, so `head_minimum_norm_gap` is kept distinct from
`head_null_fraction`.

## 3. A controlled theorem connecting superposition and aliasing

For dimension \(d\) and spread \(r\in\{1,\ldots,d\}\), let

\[
v_r=\frac{1}{\sqrt r}(\underbrace{1,\ldots,1}_{r},0,\ldots,0)^\top,
\qquad D_r=[I_d\;v_r].
\]

This dictionary represents \(d+1\) unit-norm feature directions in \(d\)
dimensions. Its mutual coherence is \(\mu(D_r)=1/\sqrt r\), and the
participation ratio of the final feature direction is
\(1/\sum_j(v_{r,j}^2)^2=r\).

**Proposition.** The map \(s\mapsto D_rs\) is injective on nonnegative
\(k\)-sparse codes if and only if \(k<r\).

**Proof.** The nullspace of \(D_r\) is one-dimensional and is spanned by
\((-v_r,1)\), whose positive support has size one and whose negative support
has size \(r\). If two nonnegative codes have the same representation, their
difference is a multiple of this vector. For a nonzero multiple, one code
must use the final feature and the other must use all first \(r\) features;
therefore both cannot be at most \(k\)-sparse when \(k<r\). Conversely, for
\(k\ge r\), the distinct codes \((0,\ldots,0,1)\) and
\((v_r,0)\) are both nonnegative, at most \(k\)-sparse, and have the same
representation \(v_r\).

To quantify the behavioral consequence, put probability \(q/2\) on each of
these two codes and probability \(1-q\) on the zero code, and let the target
be the activation of the final feature. When \(k\ge r\), the two nonzero
codes have the same representation but targets zero and one. The exact Bayes
squared risk from observing only \(H=D_rs\) is therefore

\[
\mathbb E[\operatorname{Var}(Y\mid H)]=q/4,
\]

and the aliased probability mass is \(q\). When \(k<r\), the alternative
code is outside the allowed support; the supplied distribution has no
representation collision and its Bayes risk is zero. This demonstrates that
feature overlap alone does not determine aliasing: the data support must also
contain distinct codes that the representation maps together.

The construction is implemented in `experiments/04_superposition_aliasing.py`.
It is a transparent finite-support example, not evidence about trained
networks by itself. In particular, its representation-level Bayes risk and
the MLP readout-nullspace diagnostic are distinct aliasing notions.

## 4. Statistical interpretation of simulation summaries

The per-seed outputs are the simulation units. `mean.csv` reports their mean;
`std.csv` reports the sample standard deviation with denominator \(N-1\);
`ci95.csv` reports a deterministic-seed percentile bootstrap interval for the
mean. These are pointwise intervals, not simultaneous bands over all widths,
checkpoints, or selected minima.

Featurewise random-feature sweeps reuse one generated problem per seed across
widths, so `paired_contrasts.csv` reports bootstrap intervals for per-seed
differences between adjacent widths. The MLP width comparisons share the
teacher data seed and are reported as paired by seed and checkpoint. Samplewise
random-feature sweeps regenerate each problem at each sample size; their
intervals compare mean outcomes at each sample size and are not paired
within-seed growth trajectories.

For a population prediction risk, a finite test set gives a Monte Carlo
estimate conditional on the trained model. The replicate-level intervals also
vary the generated problem and training seed. Reporting which source of
randomness is being summarized is necessary when interpreting those intervals.
