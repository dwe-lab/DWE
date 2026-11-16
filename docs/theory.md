# Estimator guarantees

Fix a prefix, current student parameters and a frozen teacher. Let
\(r=\partial L/\partial z\), with \(\mathbf1^\top r=0\). For nonzero \(r\), define

\[
m=\tfrac12\|r\|_1,\qquad a=r_+/m,\qquad b=(-r)_+/m.
\]

For any coupling \(\pi\in\Pi(a,b)\), first consider \(K\) iid pairs and set

\[
\widehat r=\frac mK\sum_k(e_{I_k}-e_{J_k}).
\]

Then \(\mathbb E\widehat r=r\). Passing this adjoint through the current head and
backbone preserves the original raw gradient in expectation. Samples and
coefficients are detached. The implementation returns the true KL scalar while
its custom first-order backward applies the sampled adjoint.

## Correct space for variance

Every fixed-amplitude coupling has the same logit mean-square error:

\[
\operatorname{MSE}_z=\frac{2m^2-\|r\|^2}{K}.
\]

For one sample, this is minimal among unbiased, zero-sum estimators with at most
two nonzero coordinates: \(\mathbb E\|X\|_2^2=\tfrac12\mathbb E\|X\|_1^2
\ge\tfrac12\|r\|_1^2\). This does not establish optimality for arbitrary
\(2K\)-sparse estimators.

For a linear head \(z=Wh+b\),

\[
\operatorname{MSE}_h(\pi)=\frac{m^2\sum_{ij}\pi_{ij}\|w_i-w_j\|^2
-\|W^\top r\|^2}{K}.
\]

Minimizing this expression within the fixed-amplitude family is squared-distance
transport between the positive and negative residuals in output-row space.
Meanwhile, the per-position head-weight error is
\(\operatorname{MSE}_W=\|h\|^2\operatorname{MSE}_z\), independent of the coupling.
These are local head metrics, not a guarantee about the full model's parameter
error, tied-embedding covariance, or Adam updates.

Independent residual pairing also improves on the specific coordinate-IS control
using \(2K\) iid draws from \(|r|/(2m)\):

\[
\operatorname{Cov}(\widehat r_{IS})-
\operatorname{Cov}(\widehat r_{independent})
=\frac{|r||r|^\top}{2K}\succeq0.
\]

It is not a claim against every optimized coordinate sparsifier. In particular,
[Efficient Unbiased Sparsification](https://arxiv.org/abs/2402.14925) and
[ATOMO](https://arxiv.org/abs/1806.04090) remain important prior work.

## Cheap geometry approximation

Order vocabulary rows by a cached principal-component score. Using the same
uniform variate in the two ordered inverse CDFs preserves their marginals. It is
optimal transport in that one-dimensional squared projected metric. It is not
necessarily optimal or even better than independence in the full hidden space.
A stale order preserves marginal correctness but loses its current-geometry
interpretation.

For positive rows \((0,0),(1,100)\) and negative rows \((0,100),(1,0)\), all side
probabilities \(1/2\), sorting by the first coordinate increases full hidden
variance from \(5000.5m^2\) to \(10000m^2\). The test suite retains this counterexample.
The real-model error–time experiment must establish whether a useful cheap order exists.

Variable-amplitude importance weighting can optimize mapped variance further,
leading to a first-order transport characterization. It requires additional
proposal/support handling and can increase head-weight noise. It is outside the
initial implementation.

## Numerical and training scope

- Forward KL uses \(r=T(p_T-q_T)\) for loss \(T^2 KL(q_T\|p_T)\). Reverse KL uses
  \(r=T p_T[\log(p_T/q_T)-KL(p_T\|q_T)]\). Only unscaled forward KL has
  \(m=TV(p,q)\).
- Positive and negative accumulated masses may differ by rounding. Each side is
  normalized and weighted by its own mass; no small nonzero residual is thresholded.
- Stable log-softmax and residuals use FP32, or FP64 for numerical tests. Native
  low-precision GEMMs and scatter reductions are not bitwise equivalent to dense backward.
- The API implements first-order differentiation. Double backward is unsupported.
- Prefixes are fixed for an update. This does not add a score-function derivative
  through an on-policy rollout distribution.
- Fixed \(K\) means approximately fixed sparse work. Smaller residual magnitude
  reduces absolute noise, not necessarily relative noise or runtime.
- Full vocabulary forward remains. The head gradient buffer and optimizer state
  remain when the head is trainable. A single accumulation buffer avoids per-chunk
  duplication, but does not make head optimization sparse.


## Residual transport and conditional learning direction

For unscaled forward KL, shared mass is c_v=min(p_v,q_v), and
p=c+ma, q=c+mb with m=TV(p,q). Residual sampling removes the already shared
mass before randomizing. It changes logits in the correction direction; it is
not a literal finite probability transfer after softmax or a parameter update.

A maximal-coupling estimator can be constructed as X=B(e_I-e_J), where
B is Bernoulli(m), independent of a residual pair with coupling pi. Conditioning
on that pair integrates out B and gives m(e_I-e_J). For any fixed linear map A,
its covariance reduction is

\[
\operatorname{Cov}(AX)-\operatorname{Cov}(Am(e_I-e_J))
=m(1-m)\mathbb E[A(e_I-e_J)(e_I-e_J)^\top A^\top]\succeq0.
\]

More generally, for K independent pairs,

\[
\mathbb E\|A(\widehat r-r)\|^2
=\frac{m^2\mathbb E_\pi\|A(e_I-e_J)\|^2-\|Ar\|^2}{K}.
\]

Marginals determine the mean; coupling determines mapped error. A=W^T gives
hidden-gradient error; A=J_theta^T gives full parameter-gradient error at the
fixed state. The implemented cached ordering approximates the former, not an
optimal full-parameter coupling. Forward-KL logit error is at most 2 TV(p,q)^2/K;
transferring this vanishing absolute-error claim to parameter space requires a
bound on the Jacobian. Relative error need not vanish.

Condition on current parameters, sampled prefixes, masks and normalization before
residual sampling. The additional raw-gradient error is a martingale difference.
For predictable learning rates and square-integrable errors, its weighted sum has
zero mean and second moment equal to the sum of weighted second moments. This is
not equality of training trajectories or of expected Adam/clipped updates.

Online OPD normally stops gradients through the student rollout distribution.
The preserved object is therefore the original conditional update field, not
necessarily the gradient of a fixed global objective. Standard fixed-objective
SGD convergence statements require a fixed state distribution and their usual
smoothness and variance assumptions; they do not automatically apply online.


## Implemented stratified coupling

The final quantile estimator uses U_k=(k-1+xi_k)/K with independent uniform
xi_k for each position and stratum. Both sides of a pair share U_k. Each stratum
has its own conditional marginals, but their average gives the original residual.
The `coupled` ablation draws iid shared quantiles using the same cached geometry.
It isolates stratification from the choice of coupling. The `independent` control
draws each side independently; `coordinate` uses 2K iid signed importance samples.
Natural-order `quantile` sampling isolates the geometry choice. Earlier runs are
identified by their archived source commit.

Let Z(U)=m A(e_I(U)-e_J(U)), g=E[Z], and mu_k=E[Z | U in stratum k]. Then

\[
E[\widehat g_{strat}]=g,\qquad
\operatorname{Cov}(\widehat g_{iid})-
\operatorname{Cov}(\widehat g_{strat})
=\frac{1}{K^2}\sum_k(\mu_k-g)(\mu_k-g)^\top\succeq0.
\]

Thus the iid formulas above are upper bounds, not equality formulas for this
implementation. The guarantee holds for any fixed linear gradient map, including
the full current Jacobian, and extends to unequal numerical side masses by using
Z= A(alpha e_I-beta e_J). It does not establish better Adam updates or task scores.
Use independent jitter per stratum, not a common random shift across strata.
Stratification is classical variance reduction, not a separate novelty claim.
