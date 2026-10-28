# GAD and learned sparse representations: mathematical results for the toy models

This is a first mathematical draft for Chapter 2. Statements labeled theorem,
proposition, or corollary below have self-contained proofs under their stated
assumptions. Numerical findings are identified separately. The proofs establish
identities, capacity separations, and counterexamples; they do not establish
that gradient descent always finds the constructions or improves aliasing.

The central explanation is that a nonlinear expansion can make information in
a compressed sparse code accessible to a linear readout. This changes the task
residual and the statistical problem of estimating that readout. GAD separates
those two effects.

Definitions follow [the GAD audit](GAD_DEFINITIONS_AUDIT.md) and
[the existing finite-design results](MATHEMATICAL_RESULTS.md). In particular,
training-nullspace error is **data insufficiency**, not generalized aliasing.

## 1. The experimental model and its three dimensions

Let z in R^p be a nonnegative latent vector. Exactly k coordinates are active,
chosen uniformly among supports of size k. Active amplitudes a are independent
Uniform[1/2,3/2], multiplied by c_k=sqrt(p/(k a_2)), where a_2=E[a^2]=13/12.
The observed input is an isometric embedding x=Jz, with J^T J=I_p.

The p-dimensional target bank is

\[
y=T(z-\mu),\qquad \mu=\mathbb E z,
\]

where T is invertible. In the experiments T is an orthogonal task mixing
followed by individual variance normalization, so each target has variance one.

The network feature map, written using column-vector conventions, is

\[
\phi(x)=\begin{bmatrix}1\\ \sigma(DE x+b)\end{bmatrix},
\qquad E:\mathbb R^{d_{\rm in}}\to\mathbb R^m,
\quad D:\mathbb R^m\to\mathbb R^w.
\]

The activation is either the identity or coordinatewise ReLU. A linear task
head predicts beta^T phi(x), with beta in R^{(w+1) x p}.

Three dimensions must be kept distinct:

- p: number of possible latent features and independent target queries;
- m: physical compression dimension;
- r: dimension of the span of the scalar readout feature functions in L2(P).

For the reported experiments p=16, m=4, w=16. An identity activation gives
r<=m+1=5. ReLU can give r=w+1=17 even at the same physical bottleneck.
The intercept contributes one function to r, but not an additional varying
coordinate to the affine subspace of output predictions.

### Proposition 1: exact moments of the active-count sweep

For p>1, the generator satisfies

\[
\mu_i=\sqrt{\frac{k}{p a_2}},\qquad
\mathbb E[z_i^2]=1,\qquad
\mathbb E[z_i z_j]=\frac{k-1}{(p-1)a_2}\quad(i\ne j).
\]

Consequently E||z||^2=p for every k. Its covariance has eigenvalues

\[
\lambda_{\parallel}=1-\frac1{a_2}=\frac1{13},\qquad
\lambda_{\perp}=1-\frac{k-1}{(p-1)a_2}
\]

in the all-ones direction and its orthogonal complement, respectively.
Both are positive for 1<=k<=p.

**Proof.** Coordinate i is active with probability k/p, and a pair i,j is
active with probability k(k-1)/(p(p-1)). Multiply these probabilities by the
corresponding scaled amplitude moments. Subtract mu mu^T to obtain covariance.
A matrix with constant diagonal and constant off-diagonal entries has the two
displayed eigenspaces. Substitution gives the eigenvalues. QED.

**Meaning.** The sweep fixes expected input energy, but does not fix the input
mean or covariance. Variance-normalizing the targets does not remove every
distributional change. A claim that isolates higher-order sparsity at fixed
covariance would require an additional controlled generator.

## 2. Exact GAD for any frozen representation

Let H_T in R^{n x d} and H_P in R^{t x d} contain row features phi(x)^T on
training and prediction inputs, where d=w+1. Let Y_T,Y_P have q target columns.
For any specified reference head beta define

\[
R_T=Y_T-H_T\beta,\qquad R_P=Y_P-H_P\beta,
\quad B=H_T^+H_T,\quad \widehat\beta=H_T^+Y_T.
\]

### Theorem 1: the frozen-map prediction identity

For every pair of designs, including rank-deficient designs,

\[
\boxed{H_P\widehat\beta-Y_P
=\underbrace{H_P(B-I)\beta}_{D:\ \text{data insufficiency}}
+\underbrace{H_PH_T^+R_T}_{A:\ \text{generalized aliasing}}
-\underbrace{R_P}_{\text{model residual}}.}
\]

There is no assumption of random weights, Gaussian data, or a particular
activation. If label noise is specified separately, its fitted contribution
H_P H_T^+ epsilon is a separate fourth term.

**Proof.** Substitute Y_T=H_T beta+R_T into H_P H_T^+Y_T, and subtract
Y_P=H_P beta+R_P. Collect terms. QED.

Write M_v=-R_P. With <U,V>_P=tr(U^T V)/(tq), squared prediction error is

\[
\mathcal R_P=\|D\|_P^2+\|A\|_P^2+\|M_v\|_P^2
+2\langle D,A\rangle_P+2\langle D,M_v\rangle_P
+2\langle A,M_v\rangle_P.
\]

The cross terms are signed and cannot generally be discarded. This identity
is relative to beta. Fitting beta on an independent reference set preserves
the identity but does not turn beta into the exact population projection.

**Terminology.** An input collision, overlap of encoder feature directions,
and the GAD vector A are different objects. In particular, a representation can
retain information that a linear readout cannot extract.

### Proposition 2: candidate heads require an estimator correction

For any other head beta_c, including a native trained head or ridge head,

\[
H_P\beta_c-Y_P=D+A+M_v+
\underbrace{H_P(\beta_c-\widehat\beta)}_{C:\ \text{estimator correction}}.
\]

**Proof.** Add H_P(beta_c-beta_hat) to Theorem 1. QED.

The risk identity now contains four energies and all six pairwise cross terms.
The A term remains the canonical **MP baseline** alias vector. Its magnitude
alone does not measure the residual leakage of the alternative estimator.
This is the convention implemented in experiment 15.

## 3. Population projection separates approximation from estimation

Condition on a frozen representation. Its representation-training data may
have been used to learn it, but the readout samples and new prediction inputs
in this section are independent of those data. Assume finite second moments.
Let beta_* minimize

\[
\mathcal M_\phi=\frac1q\mathbb E\|y-\beta^T\phi(x)\|^2,
\qquad r(x)=y-\beta_*^T\phi(x).
\]

The normal equations give E[phi r^T]=0. Write Sigma=E[phi phi^T].
Population norms below average over a new input, conditional on the frozen
representation and the readout training sample.

### Theorem 2: population risk with an optimal reference

Let C_D=(B-I)beta_* and C_A=H_T^+R_T. Then

\[
\boxed{\mathcal R_T=\mathcal M_\phi+
\frac1q\operatorname{tr}\{(C_D+C_A)^T\Sigma(C_D+C_A)\}.}
\]

The population model-residual cross terms vanish. The data/alias cross term
can remain. If the training design determines every population-visible
feature direction, then D=0 almost surely and

\[
\boxed{\mathcal R_T=\mathcal M_\phi+\mathcal A_T.}
\]

**Proof.** The prediction error on a new input is
(C_D+C_A)^T phi-r. For any fixed coefficient matrix C, the normal equations
imply E[phi^T C r]=0. Expand the squared error and apply this orthogonality.
If ker(H_T) is contained in ker(Sigma), the data vector is zero in population.
Full column rank is sufficient, but is not necessary. QED.

**Meaning.** Redundant readout coordinates may leave a nonzero coefficient
nullspace even after all population-visible functions are determined. Nullity
alone is therefore not evidence of prediction error. Also, smaller model
insufficiency can outweigh larger aliasing; no monotonic improvement of each
component is required.

Finite test averages and estimated reference heads need not have the population
orthogonality above. The experimental CSVs correctly retain their cross terms.

## 4. A sharp capacity separation at a fixed physical bottleneck

### Theorem 3: the affine output-subspace lower bound

Let y be centered, with covariance C_y and eigenvalues
lambda_1>=...>=lambda_q. Any predictor whose outputs lie in an affine subspace
of dimension at most m satisfies

\[
\boxed{\frac1q\mathbb E\|\widehat y-y\|^2
\ \ge\ \frac1q\sum_{j=m+1}^{q}\lambda_j(C_y).}
\]

In particular, this bounds every linear-compression/linear-decoding predictor.
It also bounds a linear head directly attached to m arbitrary nonlinear
features: the last linear head still restricts its output subspace.

**Proof.** For a subspace S with projector P_S, the nearest point in a+S to y
has residual (I-P_S)(y-a). Its expected squared norm is
tr((I-P_S)C_y)+||(I-P_S)a||^2. Among rank-m projectors, tr(P_S C_y) is at most
the sum of the m largest eigenvalues. To see this, diagonalize C_y; the diagonal
entries of a rank-m projector lie in [0,1] and sum to m. Allocating this mass to
the largest eigenvalues maximizes the trace. QED.

In our toy model y is an invertible affine function of the latent coordinates,
which themselves are linearly recoverable from x. Projecting y onto its top-m
covariance eigenspace is consequently an available affine rank-m predictor.
Thus the bound is attained by the population linear oracle. Since C_y is
positive definite, the bound is strictly positive whenever m<p=q.

For positive task weights W=diag(w_i), the corresponding weighted optimum is

\[
\inf_{\rm affine\ rank\le m}
\frac{\mathbb E\|W^{1/2}(\widehat y-y)\|^2}{\sum_iw_i}
=\frac{\sum_{j>m}\lambda_j(W^{1/2}C_yW^{1/2})}{\sum_iw_i}.
\]

Apply the unweighted proof to W^{1/2}y. This is the oracle calculation used for
the balanced and swapped-priority experiments.

### Theorem 4: exact one-active recovery through a three-dimensional code

For every finite p>=4, there are matrices E in R^{3 x p} and D in R^{p x 3}
such that

\[
\boxed{\operatorname{ReLU}(DEz)=z\quad
\text{for every nonnegative }z\text{ with at most one active coordinate}.}
\]

Hence the architecture with bottleneck m>=3 and expansion width w=p can have
zero population model insufficiency for the entire invertible target bank,
even though every affine rank-m predictor has the positive floor in Theorem 3
when p>m. Padding the construction with unused coordinates embeds it in m=4.

**Proof by construction.** Choose distinct angles theta_i and let

\[
v_i=(\cos\theta_i,\sin\theta_i,1)^T,\qquad E=[v_1\ \cdots\ v_p].
\]

Let rho=max_{i!=j} cos(theta_i-theta_j)<1, and choose rho<c<1. Set row i of D to

\[
d_i^T=\frac{(\cos\theta_i,\sin\theta_i,-c)}{1-c}.
\]

Then d_i^T v_i=1, while d_i^T v_j<0 for j!=i. For z=s e_j, s>=0, the j-th
preactivation equals s and every other preactivation is nonpositive. ReLU
therefore returns s e_j exactly. A final affine head
T ReLU(DEz)-T mu recovers y.
The observed embedding is handled by replacing E with E J^T. QED.

This is an existence theorem with finite, unconstrained weight norms. It is
not a theorem about the optimum of the weight-decayed training objective or
about Adam reaching these weights. It is a sufficient dimension construction,
not a claim that dimension three is minimal.

**Interpretation.** Rank-nullity prevents global linear inversion on R^p.
It does not prevent recovery on the union of one-active rays. The compressed
code itself is distinguishable on that support; the nonlinear expansion makes
the information accessible to a linear task head. It does not create new
information or invert a map on the entire ambient space.

### Corollary 4a: the nonlinear function rank can exceed the code dimension

If every coordinate is active with positive probability and amplitudes have
an interval of support, the constructed augmented features [1;z] have function
rank p+1. The preactivation matrix has rank at most three on any input set.

**Proof.** If a_0+sum_i a_i z_i=0 almost surely, restrict to z=s e_j and vary s
over its interval. This forces a_j=0 and a_0=0 for every j. QED.

Thus structural weight rank and readout function rank are different quantities.
This supplies an exact example of the distinction r=17 versus m=4 in our runs.

### Corollary 4b: co-activation introduces explicit cross-talk in this construction

For arbitrary nonnegative z, define

\[
\gamma_{ij}=\frac{c-\cos(\theta_i-\theta_j)}{1-c}>0\quad(i\ne j).
\]

The same network returns

\[
\phi_i^{\rm hidden}(z)=\left[z_i-\sum_{j\ne i}\gamma_{ij}z_j\right]_+,
\qquad z_i-\phi_i^{\rm hidden}(z)
=\min\left\{z_i,\sum_{j\ne i}\gamma_{ij}z_j\right\}.
\]

**Proof.** Expand DEz coordinatewise and use t-[t-s]_+=min(t,s) for t,s>=0.
QED. Inactive coordinates remain zero, while co-active features can suppress
one another. This formula describes this explicit construction only. Its
reconstruction loss is not automatically the optimum linear-head model term;
refitting a head can improve on the displayed decoder.

### Corollary 4c: exact recovery and finite-sample GAD have different requirements

For the exact construction the target residual is zero, so model insufficiency
and generalized aliasing are zero. Finite-sample error can still occur through
data insufficiency when the training set has not determined the task head.

Under uniform independent choices of active coordinate and continuously
distributed amplitudes, for n>=p+1 the augmented design [1;z]^T has full column
rank whenever every coordinate appears, with probability one conditional on
that event. Therefore

\[
\Pr(\operatorname{rank}(H_T)<p+1)
\le p(1-1/p)^n\le p e^{-n/p}.
\]

**Proof.** If all coordinates appear and n>=p+1, some coordinate appears at
least twice, with distinct amplitudes almost surely. A training-null affine
function then has zero intercept and zero coefficient on that coordinate;
one observation on each remaining coordinate sets all other coefficients to
zero. Bound the probability of any missing coordinate by a union bound. QED.

The learned numerical representations are approximate, not this exact
construction. Their small nonzero residuals can be amplified into aliasing.
The construction alone does not predict an aliasing peak at interpolation.

## 5. Identical preactivation spectra can give different recoverability

### Theorem 5: an orthogonal-rotation counterexample

Use the one-active distribution of Section 1 and the construction of Theorem 4
with p>3. There is an orthogonal Q in R^{p x p}
such that the maps

\[
\phi_0(z)=\operatorname{ReLU}(DEz),\qquad
\phi_Q(z)=\operatorname{ReLU}(QDEz)
\]

have identical preactivation Gram matrices on every dataset and identical
preactivation weight singular values, but

\[
\mathcal M_{\phi_0}=0,\qquad
\mathcal M_{\phi_Q}\ge\frac1p\sum_{j>3}\lambda_j(C_y)>0.
\]

Here phi_0 and phi_Q denote hidden features; an intercept is included in their
task readouts when defining the model-insufficiency terms.

**Proof.** The range of D has dimension at most three. Choose an orthonormal
basis whose first three vectors contain that range, and let Q express vectors
in that basis. The last p-3 coordinates of QDu are identically zero. ReLU
preserves those zero coordinates, so an affine readout of phi_Q has outputs
in an affine subspace of dimension at most three. Apply Theorem 3. The original
map recovers z exactly by Theorem 4. For any preactivation data matrix V,
(QV)^T(QV)=V^T V, and left multiplication by Q preserves singular values. QED.

This is a deliberately chosen rank-concentrating rotation. It does not assert
that every random rotation worsens prediction. It proves that even the whole
preactivation Gram matrix, together with the task distribution, is insufficient
to determine post-ReLU linear recoverability without activation-axis geometry.

The random rotations in experiments 14–15 are empirical counterparts, not
instances claimed to attain this worst-case bound. Their activity-matched
variants change biases and therefore do not preserve the biased Gram matrix.

## 6. Residual direction and finite-sample amplification

Work in an orthonormal basis for the population-visible feature functions so
that Sigma=E[phi phi^T] is positive definite of size r x r. This removes exact
redundant coordinates without changing the MP predictor. Define

\[
\widehat\Sigma_n=\frac1n\sum_{i=1}^n\phi_i\phi_i^T,
\qquad \Xi_n=\frac1n\sum_{i=1}^n\phi_i r_i^T.
\]

### Theorem 6: exact residual-covariance expression

Whenever the readout design has full column rank,

\[
\widehat\beta-\beta_*=\widehat\Sigma_n^{-1}\Xi_n,
\qquad
\boxed{\mathcal A_n=\frac1q\operatorname{tr}
\left(\Xi_n^T\widehat\Sigma_n^{-1}\Sigma
\widehat\Sigma_n^{-1}\Xi_n\right).}
\]

**Proof.** Substitute Y_T=H_T beta_*+R_T into the full-rank least-squares
solution. Since H_T^+R_T=Sigma_hat^{-1}(H_T^T R_T/n), integrate its prediction
energy over a new independent input. QED.

**Meaning.** A small empirical covariance eigenvalue can amplify a residual
component that aligns with its direction. The residual-feature correlation
Xi_n matters alongside conditioning. Changing the representation changes
Sigma_hat, the residual, and their alignment simultaneously. A pseudoinverse
norm alone is not the realized aliasing penalty.

### Proposition 3: the large-sample residual leverage law, with explicit conditions

Assume iid readout samples, fixed r, Sigma positive definite,
E||phi||^2<infinity and E[||phi||^2||r||^2]<infinity. Let Z be the centered
Gaussian r x q matrix with covariance equal to that of vec(phi r^T). Then

\[
n\mathcal A_n\ \Rightarrow\ \frac1q
\operatorname{tr}(Z^T\Sigma^{-1}Z).
\]

Here the full-rank formula is used on the event that Sigma_hat is invertible;
that event has probability tending to one. If, **in addition**, n A_n is
uniformly integrable, then

\[
\mathbb E\mathcal A_n=\frac{\kappa_\phi}{n}+o(n^{-1}),\qquad
\kappa_\phi=\frac1q\mathbb E\left[\|r\|^2\,
\phi^T\Sigma^{-1}\phi\right].
\]

**Proof.** The population normal equations make E[phi r^T]=0. Apply the
multivariate CLT to sqrt(n) Xi_n, the law of large numbers to Sigma_hat, and
the continuous mapping theorem to Theorem 6. Uniform integrability permits
convergence of expectations. The Gaussian quadratic-form expectation is
tr(Sigma^{-1} E[phi phi^T ||r||^2])/q, giving kappa_phi. QED.

If E[||r||^2 | phi]=q M_phi, this coefficient simplifies to r M_phi because
E[phi^T Sigma^{-1}phi]=r. That conditional-energy assumption is not automatic
for deterministic, misspecified toy tasks.

Uniform integrability is also not automatic: rare nearly singular designs
can invalidate an expected-risk expansion even when a typical-sample limit
holds. The experiments do not verify this assumption. Accordingly, a universal
expected 1/n law or a sharp double-descent curve is not claimed here.

## 7. Controlled perturbations measure a separate sensitivity

### Proposition 4: exact imposed-noise response

For a frozen map and MP readout, put K=H_P H_T^+. A training-target perturbation
Delta changes predictions by K Delta. If its entries are independent, have
mean zero and variance sigma^2, then

\[
\boxed{\mathbb E_\Delta\frac{\|K\Delta\|_F^2}{tq}
=\frac{\sigma^2}{t}\|K\|_F^2.}
\]

The worst ratio between prediction RMS and training-perturbation RMS is

\[
\boxed{\sup_{\Delta\ne0}
\frac{\|K\Delta\|_F/\sqrt{tq}}{\|\Delta\|_F/\sqrt{nq}}
=\sqrt{n/t}\,\|K\|_2.}
\]

**Proof.** Linearity gives the response. Since E[Delta Delta^T]=q sigma^2 I,
take a trace to obtain the expected energy. The operator norm gives the upper
bound for the RMS ratio, attained by a leading right singular vector in one
output column. QED.

For a new prediction input and a full-rank training design, the conditional
population noise energy is sigma^2 tr(Sigma(H_T^T H_T)^{-1}). Consequently,
under the law-of-large-numbers assumptions above,

\[
n\,\mathcal R_{\rm imposed\ noise}(T)\longrightarrow\sigma^2 r
\quad\text{almost surely}.
\]

**Proof.** Rewrite the trace as sigma^2 tr(Sigma Sigma_hat^{-1})/n. QED.
This is a limit of the noise-averaged risk conditional on training inputs;
it is not automatically a limit of expectations over ill-conditioned designs.

Thus a larger readout function space has an explicit large-sample imposed-noise
cost. The linear and ReLU maps can have different r at the same physical m.
This does not give a universal finite-sample ordering of their sensitivities.

K is a readout propagation operator. The full GAD omitted-feature operator
requires an omitted feature matrix U_T and is K U_T. Shared Gaussian probes,
shared structured contamination, and map-dependent residualized probes must
therefore retain their distinct interpretations.

## 8. What the numerical results add, and what they do not prove

The following findings refer to source job 14814813 and mechanism-check job
14822110, with 24 independent representation seeds. Readout draws, rotations
and probe directions were averaged within seed before paired intervals.

| Finding | Numerical evidence | Mathematical connection |
| --- | --- | --- |
| Sparse nonlinear recovery | At k=1, original learned ReLU has MSE about 0.000383 at n=128; the linear oracle has about 0.745. | Theorems 3–4 establish a capacity separation in this architecture class; they do not identify the learned weights with the construction. |
| Encoder learning contributes | At k=4 and n=4096, fresh joint training has MSE 0.4544 versus 0.6574 for a frozen random encoder with learned decoder. Joint training wins in all 24 seeds. | A static architecture/optimizer observation. No theorem here says Adam finds a globally optimal encoder. |
| Gate axes matter | At k=4,n=128, original gates have MSE 0.5210, rotations 0.7630, activity-matched rotations 0.7658. | Theorem 5 proves spectra alone cannot determine recovery; the observed random-rotation effects remain empirical. |
| Approximation and aliasing can move in opposite directions | At k=4,n=128, learned linear has model term 0.7164 and alias term 0.0285; learned ReLU has 0.4499 and 0.0712. | Theorems 1–2 permit lower total error with larger aliasing. |
| Some amplification improvements are real | At k=4,n=128, isotropic perturbation energy is 0.1623 for original gates versus 0.1768 after rotation; paired increase 0.0145, CI [0.0082,0.0213]. | Proposition 4 identifies this measurement exactly. It is not uniform noise suppression: learned linear gives about 0.0408. |
| Dense-data failure is readout instability | At k=16, MP MSE is 21.84 at n=128, ridge gives 0.796, and MP at n=4096 gives 0.752. | Theorems 1 and 6 expose residual propagation through ill-conditioned readouts; Proposition 2 handles ridge without renaming GAD terms. |

The fixed-random-encoder arm and fresh joint arm have matched starts and update
counts. The frozen-learned-encoder arm has extra pretraining and is a secondary
diagnostic. Original native heads saw 2048 representation-training examples;
they should not be treated as matched-data alternatives to a 128-example probe.

Reference estimates are adequate for the large approximation comparisons but
not exact. For k=4, increasing the reference from 16384 to 65536 changes model
insufficiency from 0.450270 to 0.449936. At readout n=4096 it changes the small
alias term from 0.002196 to 0.001881. Exact constants for tiny alias terms need
more reference precision. The population projection is used in the population
proofs; the reported experiments use estimated reference heads.

The evidence does not establish strict Marchenko–Pastur spectra, calibration of
biases to a noise edge, universal ReLU spectral filtering, or preferential
protection of decision boundaries. No claim of novelty relative to the full
literature is made by these self-contained proofs.

The exact sparse capacity separation above is proved for k=1. No sharp optimum
recovery curve for k>=2, guarantee of a random frozen encoder's recoverability,
or optimization guarantee for the observed jointly learned maps is proved here.
Those observed advantages remain numerical findings. The co-activation formula
describes one explicit construction and is not a substitute for such theorems.

## 9. Proposed Chapter 2 mathematical narrative

1. Define the frozen-map GAD identity and distinguish physical compression,
   function-space rank, input collisions, and realized residual aliasing.
2. Prove the affine rank-m task-bank floor and construct exact sparse recovery
   through a three-dimensional code with nonlinear expansion.
3. Prove that orthogonal preactivation rotations can preserve every Gram value
   while changing post-activation recoverability.
4. Derive the residual-covariance expression for aliasing and identify the
   separate effects of approximation, finite-sample conditioning, and alignment.
5. Use the controlled perturbation identities and empirical readout curves to
   describe the estimation cost of the richer accessible function space.

Training dynamics and guarantees about the geometry selected by gradient
descent are subsequent questions. The present results concern fixed maps and
the existence or measurement of their static properties.
