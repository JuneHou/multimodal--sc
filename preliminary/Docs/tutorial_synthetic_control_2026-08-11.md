# Synthetic control, the solver, and why our experiment worked — a tutorial

2026-08-11. Written for Jun, using this project's own results as the running
examples. Reading list with specific sections at the end. Companion:
[trial_dataset_decision_2026-08-10.md](trial_dataset_decision_2026-08-10.md)
(reports 11–14 hold the numbers cited here). §5–6 added 2026-08-12 from the
meeting-prep Q&A: the theory behind the experimental grid (K, wstar, m,
sigma) and behind the recovery-vs-generation split.

## 1. The problem synthetic control solves

Causal inference's core object is the **potential outcome**: patient i has
two futures, Y_i(treated) and Y_i(untreated), and we only ever observe one.
The unobserved one is the **counterfactual**. Every causal method is a
strategy for filling it in.

Synthetic control (SC) fills it in with a *weighted combination of other
units*. Setting: a panel — units i (states, firms, patients), time t, an
outcome Y_it observed for everyone before the treatment time T0, and we want
unit 1's untreated future after T0. SC says: find weights w over the
untreated "donor" units such that the weighted donors reproduce unit 1's
**pre-treatment history**, then carry those weights forward:

    min_w  Σ_{t ≤ T0} ( Y_1t − Σ_j w_j Y_jt )²
    s.t.   w_j ≥ 0,   Σ_j w_j = 1                      (the simplex)

    counterfactual:  Ŷ_1t(untreated) = Σ_j w_j Y_jt   for t > T0
    treatment effect: τ_t = Y_1t(observed) − Ŷ_1t

That is the entire estimator. Two constraints — nonnegativity and
sum-to-one — look innocent but carry most of the method's character:

1. **No extrapolation.** The synthetic unit lives inside the convex hull of
   the donors: it is an average of real units, never a leveraged
   combination (contrast regression, which happily uses coefficients of
   −3.7 and +4.2). You can only be "made of" patients that exist.
2. **Sparsity for free.** Optima of a least-squares objective over a
   polytope land on low-dimensional faces: typically only a handful of
   donors get nonzero weight. Abadie & L'Hour prove the penalized solution
   uses at most p+1 donors (p = number of matching dimensions). This is why
   SC weights are *readable*: "California ≈ 0.33 Utah + 0.23 Nevada + …".
3. **Interpretability is the causal claim.** The weights are not tuning
   parameters — they are the statement "these specific units, in these
   proportions, are the comparison." That is what your advisor means by
   "build a causal model, then impute the counterfactual": the weight
   vector *is* the model; the imputation is one matrix multiply.

Origin story to know: Abadie & Gardeazabal 2003 (Basque terrorism),
Abadie, Diamond & Hainmueller 2010 (California tobacco law — the canonical
paper), since called "arguably the most important innovation in the policy
evaluation literature in the last 15 years" (Athey & Imbens).

## 2. Why is this *causal*? The latent factor model

Matching pre-treatment trajectories would be mere curve-fitting unless
something connects "fits the past" to "would have matched the future." The
standard assumption is a **linear latent factor model**:

    Y_it = δ_t + λ_i · f_t + ε_it

Each unit has a fixed latent loading vector λ_i ("what kind of unit this
is"); each period has a factor vector f_t ("what happened in the world");
outcomes are their inner product plus noise. The key theorem (ADH 2010): if
the weighted donors match unit 1's pre-period *long enough and well
enough*, they must have matched its **loadings** λ_1 — you can't fake a
long trajectory match without being the same kind of unit — and units with
the same loadings respond identically to future factors. So pre-period fit
transfers to post-period counterfactual validity. The bias shrinks as the
pre-period grows relative to the noise.

Three refinements that matter to us:

- **Ferman (JASA 2021): matching loadings ≠ recovering weights.** With many
  donors, SC can reconstruct λ_1 exactly while the weight vector itself is
  non-unique (many recipes cook the same dish). *Reconstruction* and
  *weight identification* are different events — which is why our
  experiment scored them separately, and why the uniqueness check was
  needed before trusting the exact m=all recovery.
- **Abadie & L'Hour (JASA 2021): the polytope.** If the target sits in the
  *interior* of the donor hull, infinitely many exact-fit weight vectors
  exist. Our check found the opposite geometry: each mixed target sits on
  the hull *boundary*, in the face spanned by its true donors — that is
  why its recipe was unique (LP-certified, off-support mass ≤ 1.7e-12).
- **Doudchenko & Imbens 2016: the constraints are choices.** Drop
  sum-to-one, add an intercept, allow negative weights — you get a family
  of estimators trading interpretability against fit. (Our observable-phase
  `fit.py` implements their intercept; the trial run kept classical
  constraints.)

## 3. The solver — what `solve_simplex_qp` actually does

The optimization is a **quadratic program (QP)**: quadratic objective,
linear constraints. Expanding the objective gives the Gram form

    f(w) = wᵀ G w − 2 bᵀ w + c,   G = A Aᵀ (donor similarities),
                                   b = A x  (donor–target similarities)

— note the data enters only through inner products, which is why our code
precomputes G once (m×m) instead of touching the full (T0+1)·d-dimensional
vectors per iteration; that is the difference between seconds and hours.

The constraint set (the simplex) is a polytope. The solver, SLSQP
(Sequential Least-Squares Quadratic Programming), walks the feasible set:
at each step it solves a local QP approximation, projects onto the
constraints, and stops when the **KKT conditions** hold — the
first-order-optimality certificate for constrained problems, which in plain
terms says: *the objective's downhill direction points straight out of the
feasible set, so no feasible move improves anything*. For a convex QP over
a convex set, KKT ⇒ global optimum: there are no local-minimum traps. This
is why "the solver found it" and "it is the best possible" coincide here —
completely unlike neural network training.

**When is the optimum unique?** Two sufficient conditions, and our
experiment exercised both:
- *Strict convexity*: if G is full-rank (donors ≤ dimensions, in general
  position), the bowl has one bottom. Our m=50 cells: 50 donors in 768
  dims, minimum eigenvalue 0.018–0.093 → trivially unique.
- *Constraint geometry*: with more donors than dimensions (m=all: 1,263 in
  768), G is rank-deficient and the bowl has a flat valley — a
  494-dimensional affine set of exact fits. Uniqueness must then come from
  the inequality constraints: w ≥ 0 cuts the valley down to a single
  point when the target lies on a low-dimensional face of the hull. That is
  a *sparse recovery* phenomenon, the same mathematics as compressed
  sensing: a K-sparse nonnegative solution is unique when the donors are
  "sufficiently scattered" (Fu & Gillis). Our LP certificate verified it
  empirically rather than assuming it.

**Why exact recovery isn't leakage** (your question, kept for the record):
there is no learned model and no train/test split. Existence of a perfect
fit is trivial by construction; the scientific content is *uniqueness*
(certified by an independent LP solver) and *falsifiability* (the same test
failed on digit-string embeddings — a rigged test cannot fail). Fields
built on inverse problems — compressed sensing, matrix completion,
hyperspectral unmixing — publish exact recovery as the expected behavior in
the noiseless identifiable regime and put the science at the regime's
boundary. So do we: the headline is 0.884-under-realistic-retrieval, and
the exact cell is the identifiability certificate behind it.

## 4. The latent twist — what this project adds

Classical SC runs in *outcome space*: Y_it is GDP, cigarette sales, a lab
value. Our move: the unit-time observation is a high-dimensional artifact
(a clinical note; later, a satellite image), and SC runs in a **learned
embedding** of it. Everything in sections 1–3 carries over mechanically —
same QP, same simplex — but every guarantee now depends on the *embedding
geometry*, which nobody certifies for you. Hence the battery we ran:

| classical concept | our experiment | result |
|---|---|---|
| does a valid comparison exist? | oracle/m=all planted recovery | exact, LP-unique — the latent supports identification |
| donor selection | retrieval (m=50) | the only error source (2.7/5 true donors found) |
| weight identification | dominant-weight (0.9) recovery | 0.884 — supported |
| representation validity | H2 code-space vs text-space weights | disagree at chance — representation choice is load-bearing |
| counterfactual generation | L3/L4 decode of mixtures | stand-in decoder copies, can't interpolate — needs the trained decoder |

The research program in one sentence: classical SC assumes the space in
which mixing is valid; we *test* for it now and will *learn* it next —
retrieval, latent, and decoder as one system, with the planted-mixture
battery as the validity instrument.

## 5. The grid as identification theory — what K, wstar, m, sigma each isolate

The H1 experiment is a factorial grid, and each knob is not a hyperparameter
but a *piece of identification theory made testable*. The design principle:
total recovery error is a sum of distinguishable failures, and each knob
turns exactly one of them on or off.

**K (number of planted donors) — the sparsity level.** In sparse-recovery
language, K is the sparsity of the true solution. Identifiability of a
K-sparse nonnegative solution degrades as K grows relative to how
"scattered" the dictionary is (Fu & Gillis): a 2-donor blend sits on an edge
of the hull, easy to pin down; a 10-donor blend sits deep in a
high-dimensional face, with many more candidate imitations nearby. Sweeping
K ∈ {2, 5, 10} traces this difficulty curve empirically.

**wstar (the planted weight pattern) — two identifiability regimes, not two
arbitrary test cases: the **dominant-weight** family and the **diffuse**
family.** The pattern controls *where the information about each donor
lives*:

- **dominant-weight** (w* = 0.9 on one donor): the blend is geometrically
  *close to its dominant donor*. For unit-norm donors, cos(z*, z_j) grows
  with w_j; with 0.9 planted, the dominant donor's cosine to the target
  approaches 1 while the 0.025 donors sit near the corpus-typical
  similarity (~0.71 in n2c2) and disappear into the crowd. This regime
  stresses **near-twin confusion**: the solver must distinguish the true
  dominant donor from other records that also look like the target, and
  size its weight. It is the advisor's "0.9" case and the regime closest to
  how SC is used in practice (a patient dominated by one comparator).
- **diffuse** (all K weights drawn at random, spread across the donors with
  no dominant one — uniformly over the simplex — e.g.
  0.27/0.23/0.32/0.01/0.17): no donor is close to the blend — *every*
  donor's cosine to the target is moderate, so each one can be out-ranked
  by unrelated records that happen to be globally similar. This regime
  stresses **faint contributions drowning**. It is graded by L1 over the
  whole weight vector, which is why it is the harder family by
  construction.

The split in the results (dominant-weight supported, diffuse missed) is
therefore not noise — it is the geometry above showing up on schedule:
cosine retrieval always surfaces a 0.9 donor (top-1 = 100%) and
systematically misses 0.17 donors (2.7 of 5 found). A useful mnemonic: *the
weight a donor carries is also the signal strength by which retrieval can
find it.*

**m (candidate-pool size) — the error decomposition.** Sparse-recovery
theory splits estimation into **support recovery** (find which coordinates
are nonzero) and **estimation on the support** (size them). Our three pool
settings realize exactly that decomposition as an ablation:

- `oracle` hands over the true support → tests estimation alone (L1 ~1e-5:
  estimation is essentially perfect);
- `m = all` guarantees the support is *present* but not *identified* →
  tests whether the geometry can find it among 1,263 candidates (exact,
  LP-certified unique: it can);
- `m = 50` makes a retrieval step propose the support → the only setting
  where error appears, so **all observed error is support-proposal error**.

This is the logical skeleton of the headline claim "the causal geometry is
sound; retrieval is the bottleneck" — it is not an interpretation, it is a
subtraction.

**sigma (noise on the target) — the resolution floor.** A strictly convex
QP's solution is Lipschitz-stable in perturbations of the target, so small
noise moves the recovered weights proportionally. The interesting question
is calibration: *how much* noise is realistic? Our nonzero sigma is matched
to the residual a real patient leaves when fitted with 50 donors — i.e., we
perturb the planted blend by exactly the amount by which real patients fail
to be blends. Recovery under that sigma bounds what weight precision one may
ever claim on real data; recovery at sigma = 0 is the identifiability ideal
the noise erodes.

## 6. Recovering the recipe vs cooking from it — why generation is a separate problem

A question that came up repeatedly in rehearsal: *what is "generated" in
this experiment, and how does it relate to the 0.884?* The clean answer is
that counterfactual generation decomposes into two stages with completely
different mathematical character, and the battery tested them separately.

**Stage 1 — the inverse problem (H1): find the recipe.** Nothing is
generated here in the LLM sense. The "fake patient" is a *vector*
z* = Σ w*_j z_j, produced by arithmetic; the LLM's only role is the encoder
that supplied the coordinates z_j. Recovery is done by the convex solver of
§3 — deterministic, certificate-bearing, no training. The 0.884 lives
entirely in this stage: it says the recipe is findable (and, with the true
donors present, provably unique). Division of labor worth stating in one
line: *the LLM provides the geometry; classical optimization does the
inference.* Every guarantee in §1–3 routes through the embedding geometry
and none through the LLM's generative ability.

**Stage 2 — the forward problem (H3/H4): cook from the recipe.** To produce
a counterfactual *note*, the blended vector must be mapped back to text.
Here the mathematics changes character: convex combination is an operation
on vector spaces, and **token sequences are not a vector space** — there is
no "0.5 × note A + 0.5 × note B" defined on surface text. Interpolation has
to be *defined through the latent*: a faithful decoder D should satisfy
(i) cycle consistency — embedding D(z_α) lands back near z_α — and
(ii) attribute monotonicity — clinical severity of D(z_α) moves smoothly
with α. Our stand-in decoder (an instruction-tuned LLM given the donor
notes and weights in a prompt) passed (i) and failed (ii).

Why an off-the-shelf LLM fails (ii) is itself explainable, not mysterious.
At temperature 0 the model emits the single highest-probability
continuation; conditioned on two source documents, the mode of that
distribution is anchored to *one* of them — greedy decoding performs
**selection, not averaging** (softmax picks, it does not blend).

What "copying" means here, precisely: the diagnostic is the share of the
generated note's **word 8-grams** (every 8-word window) that occur verbatim
in a given source note. Two notes with no lifting share essentially no
8-grams — the internal calibration is the diffuse family itself, where the
generated note's median overlap with its *top-weight* donor is 0.00 even
though that donor contributed ~0.3 of the blend. Against that zero
baseline, an overlap of 0.65 means most of the generated text was lifted
verbatim from that one donor. The gap to 1.0 is largely mechanical, not
evidence of blending: generation was capped at 800 tokens while the median
real record is ~1,022, donor notes were truncated in the prompt, and the
model lightly paraphrases — even at α = 1.0, where the task is to render
*one* patient with no mixing at all, overlap reaches only 0.75 (and 0.95
for the other endpoint, α = 0).

The observed behavior matches the selection account in detail across both
experiments:

- when one weight is overwhelming (dominant-weight 0.9; α ≥ 0.75), *which*
  donor gets lifted from follows the weight — 0.65 median overlap with the
  0.9 donor in H3, 0.75 with the α = 1.0 donor in H4, against the 0.00
  no-lifting baseline;
- when weights are close (diffuse; α = 0.25–0.5), the model still lifts
  most of its text from *some single* donor (pooled overlap median 0.84 —
  84% of its 8-grams occur in at least one donor) but the choice decouples
  from the weights (median overlap with the top-weight donor: 0.00) and
  falls to weight-irrelevant features of the prompt — in H4, where donor
  order was fixed, every ambiguous α was resolved in favor of the
  *first-listed* donor, and the copy-identity switch landed at α ≈ 0.75
  rather than 0.5, exactly the signature of a position bias the weight must
  overcome. (Inference from the fixed order; the order-swapped control was
  not run.)

So the two stages have opposite verdicts and that is the point: the recipe
is recoverable to certificate precision, while cooking from the recipe has
no mechanism in a prompted LLM — the decoder must be *trained* so that
continuity in z (which stage 1 certifies is meaningful) is inherited by the
text it emits. That trained decoder, together with learned retrieval (the
§5 bottleneck), is the proposed contribution.

## 7. Reading list — in order, with the specific sections

**Start here (survey + the one book chapter):**
1. **Abadie, "Using Synthetic Controls: Feasibility, Data Requirements, and
   Methodological Aspects," *J. Economic Literature* 59(2), 2021.** The
   canonical modern survey by the method's inventor. Read §2 (the
   estimator), §3 (what makes it credible — the factor model discussion),
   §5 (practical requirements: pre-period length, donor pool discipline).
   Maps directly onto every design choice in our `fit.py`/MODULE.md.
2. **Scott Cunningham, *Causal Inference: The Mixtape*, ch. 10 (Synthetic
   Control).** Free at mixtape.scunning.com. The gentlest complete
   walkthrough, with the California tobacco example coded out. Read before
   or alongside Abadie if the JEL prose is dense.

**The three theory papers our experiment leans on:**
3. **Abadie, Diamond & Hainmueller, JASA 2010** — §2 for the factor model
   and the bias bound (the "pre-period fit ⇒ loadings match" argument in
   §2.3 is the heart of SC's causal claim).
4. **Abadie & L'Hour, JASA 2021** — §2–3: non-uniqueness when the target is
   hull-interior, the penalization fix, the ≤ p+1 sparsity theorem. This is
   the paper our uniqueness check answers to.
5. **Ferman & Pinto, *Quantitative Economics* 2021** (+ Ferman JASA 2021)
   — imperfect pre-treatment fit; the loadings-vs-weights distinction; why
   "reconstructs the target" and "recovers w" must be reported separately.

**The ML-flavored view (most natural for your background):**
6. **Amjad, Shah & Shen, "Robust Synthetic Control," JMLR 19(22), 2018** —
   SC as denoising + regression on a low-rank panel; written for an ML
   audience; §2–3. Their spectral view is the bridge to embeddings.
7. **Athey, Bayati, Doudchenko, Imbens & Khosravi, "Matrix Completion
   Methods for Causal Panel Data Models," JASA 2021** — counterfactual
   imputation as matrix completion; read §1–2 for the unifying frame
   ("SC, DiD, and matrix completion are one family").
8. **Doudchenko & Imbens, NBER w22791, 2016** — §3: which constraints to
   keep (intercept, sum-to-one, nonnegativity) and why; the menu our
   observable-phase extensions came from.

**The optimization background (targeted, not cover-to-cover):**
9. **Boyd & Vandenberghe, *Convex Optimization*** (free PDF at
   stanford.edu/~boyd/cvxbook): §4.4 (quadratic programs — our problem
   class), §5.5.3 (KKT conditions — the optimality certificate), §2.1–2.2
   (convex sets, simplex, polyhedra). ~40 pages total; everything the
   solver does becomes obvious afterwards.
10. **Candès & Wakin, "An Introduction to Compressive Sampling," IEEE SPM
    2008** — the friendliest entry to exact sparse recovery and why
    noiseless exactness is a theorem, not a bug. Read §1–2 only.
    Deeper/optional: Fu & Gillis, arXiv 2007.11446 (the
    sufficiently-scattered identifiability condition — the formal version
    of why our m=all recovery was unique).

**Closest neighbors to the project itself (related-work core):**
11. **Qian et al., "SyncTwin," NeurIPS 2021** — per-patient synthetic
    control with a learned temporal representation, structured EHR only;
    the single closest prior method. Note their pre-treatment fit *gate*
    (δ=0.12 → 75% acceptance) — a practice we should adopt.
12. **Arora, Li, Liang, Ma & Risteski, TACL 2018** — linear structure of
    word senses; their Theorem 2 is our mixing assumption proved in a text
    embedding space; they ran the planted-mixture experiment forward,
    we inverted it.
13. **Iordache, Bioucas-Dias & Plaza, "Sparse Unmixing of Hyperspectral
    Data," IEEE TGRS 2011** (+ the HySUPP survey, TGRS 2024, for modern
    tooling) — the same simplex-weights-from-a-library problem in
    *satellite imagery*, with the planted-abundance validation protocol as
    field standard. Our future domain already speaks this language.
14. **Liu, Wang & Xu, AJPS 2024** ("A Practical Guide to Counterfactual
    Estimators") — the umbrella your advisor's vocabulary comes from;
    skim for the diagnostics (placebo tests, equivalence tests) we should
    mirror in the causal phase.

**Suggested order for a two-week pass:** 2 → 1 → 9(§4.4, §5.5.3) → 3 → 4 →
5 → 6 → 11 → 12 → the rest as needed. After items 1–5 you can re-read
reports 11–14 in the decision doc and every number should feel inevitable.
