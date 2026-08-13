# How I designed and implemented the experiment — a step-by-step walkthrough

2026-08-11. Rehearsal script for explaining the preliminary study to anyone,
including a listener with **no causal-inference background**. Every step:
what I did, why, a concrete example, and the one sentence to say. Companion
to [FINAL_REPORT_latent_recovery_2026-08-11.md](FINAL_REPORT_latent_recovery_2026-08-11.md)
(which holds the full numbers).

---

## Step 0 — What problem is this? (for a listener with zero causal background)

Start with an everyday example. You take a pill and your headache goes away.
Did the pill work? To answer, you'd need to see the *other* you — the one
who didn't take the pill. That missing observation is called the
**counterfactual**, and it is the central problem of causal inference: for
any treated person, the untreated version of them is unobservable.

Medicine's classical answer is the randomized trial: average over thousands
of people so the two groups stand in for each other's counterfactuals. But
that only answers "does the pill work *on average*." Our project wants the
individual answer: *what would THIS patient's next months have looked like
without the treatment?*

**Synthetic control** is a method for that. Idea: build a "stand-in twin"
for the patient as a **weighted blend of other real patients** (called
**donors**) — chosen so the blend matches the patient's own history. If
0.9 × patient A + 0.1 × patient B tracks everything about you so far, then
A and B's *later* records, blended 0.9/0.1, are a reasonable stand-in for
your untreated future. The weights (0.9, 0.1) are the whole model: they say
*who you are like, and how much*.

> **Say this:** "The counterfactual is the missing branch of an A/B test
> with n=1. Synthetic control fills it in with a weighted blend of similar
> real patients — and the weights are the model. My experiment asks whether
> those weights can be trusted."

## Step 1 — Turn "trust the weights" into a falsifiable test

Problem: on a real patient, nobody knows the *true* weights — there is no
answer key, so you can't grade the method. My move: **create the answer key
myself.**

Toy example with 3-dimensional vectors (real ones are 768-dimensional):

    patient A (severe cardiac):  a = [1, 0, 0]
    patient B (mild diabetes):   b = [0, 1, 0]

    I secretly build a fake patient:  z = 0.9·a + 0.1·b = [0.9, 0.1, 0]

    I hand the solver only z and a pile of candidate patients,
    and ask: whose blend is this, and in what proportions?

    Correct answer: 90% A, 10% B. Anything else is a measurable error.

This is called a **semi-synthetic** or **planted** design: the inputs are
real (real patients' records), but the mixture is manufactured so the truth
is known. Fields that live on inverse problems — compressed sensing, image
unmixing in satellite imagery — validate exactly this way, because it is the
only place ground truth exists.

> **Say this:** "You can't grade weight recovery on real patients — no
> answer key. So I manufacture patients whose recipe I know, hide the
> recipe, and grade the solver on finding it."

## Step 2 — Freeze the grading rules before running (pre-registration)

Before any experiment ran, I wrote down — in a dated document I then never
edited — the four hypotheses, the exact numeric pass/fail thresholds, the
random seeds, and rules for interpreting each outcome combination.

Example of a frozen rule: *"H1 is supported if the recovered dominant
weight is within ±0.10 of the planted 0.9 AND the dominant donor is
identified in ≥90% of runs; refuted if the error exceeds 0.25."* When I
later deviated (I trimmed one over-expensive control mid-run), the
deviation and its reason were written into the results file.

Why: with thresholds fixed in advance, I cannot unconsciously move the
goalposts after seeing numbers. This is standard in clinical trials and
increasingly in ML research.

> **Say this:** "All thresholds were fixed in writing before the first run.
> Ten-ish deviations happened; every one is recorded with its reason."

## Step 3 — Choose data by verified properties (and the failure that taught me)

The experiment needs **real clinical text, at several timepoints per
patient**. My first dataset choice — a breast-cancer clinical trial —
passed every check on paper, then failed the one check I hadn't done:
looking at the actual text. Its "medication names" were raw billing codes:

    what I expected:  "Medications: cyclophosphamide; ondansetron."
    what it was:      "Medications: 00955301001; 00003032."

Text encoders see digit strings there, not medicine — every record looks
like every other record (their embedding similarity was 0.99 across the
board). I declared that run invalid and rebuilt on the **n2c2 longitudinal
corpus**: 288 real patients, on average 4.4 genuine clinical notes each
spanning ~4 years — e.g., a cardiology follow-up letter, then a clinic
note about diabetes management, then a consult about an amputation site.
(I already held this dataset under a signed data-use agreement.)

The invalid run was kept, deliberately, as a control: it proves the test
*can* fail — so its later success means something.

> **Say this:** "My selection rule now: never adopt a dataset without
> reading an actual text sample and counting timepoints per patient. I
> learned that by burning a day on a dataset whose 'text' was billing
> codes — and that failed run became my negative control."

## Step 4 — The pipeline, four scripts

1. **Parse.** Each patient's file → their sequence of dated notes. Detail
   that mattered: two patients' notes were stored out of chronological
   order, so I sort by the parsed date, not file order. Every count is
   asserted against an inventory built by an independent pass — the script
   crashes loudly on any mismatch rather than proceeding.
2. **Embed.** Each note → a 768-dimensional unit vector using a text
   encoder (Qwen3-Embedding-8B). For a non-NLP listener: an embedding is a
   coordinate system where similar meanings sit near each other — the
   note "renal cell carcinoma, radiation planned" lands near another
   patient's "incidentally found renal mass" note. I verified exactly that
   kind of neighborhood by reading top-3 neighbors for sample patients.
   Also verified: encoding the same note twice gives bit-identical vectors.
3. **Plant and solve.** Build z = Σ wⱼ·(vector of note j) for chosen
   weights; then solve:

       minimize  ‖z − Σⱼ wⱼ·zⱼ‖²   subject to  wⱼ ≥ 0,  Σⱼ wⱼ = 1

   The constraints mean: the answer must be a *blend* (no negative
   ingredients, proportions summing to 100%). This is convex optimization —
   a bowl-shaped problem where the solver's stopping condition is a
   mathematical certificate of global optimality. **There is no training
   and no learned model anywhere in this loop.**
4. **Score.** Weight error (sum of |recovered − true| over donors),
   dominant-donor hit rate, confidence intervals from 2,000 bootstrap
   resamples; 500 planted patients per experimental condition.

> **Say this:** "Four scripts: parse, embed, plant-and-solve, score. The
> solver is a convex program with an optimality certificate — nothing is
> trained, so nothing can leak."

## Step 5 — Controls: every claim has a way to die

- **Oracle gate.** First, hand the solver a pool containing *exactly* the
  true donors. If it can't recover the recipe even then, stop everything.
  (It recovered to error ~0.00001.)
- **The all-donors control.** Pool = all 1,263 records, so the true donors
  are guaranteed present. This separates two failure stories: "the search
  didn't find the right donors" vs "the geometry can't identify weights."
- **Uniqueness certificate.** Toy version of the worry: if a third patient
  C = [0.7, 0.3, 0] existed, then 0.9A + 0.1B and some blend involving C
  might fit the target *equally well* — and the solver's choice between
  them would be arbitrary. With 1,263 candidates in 768 dimensions this is
  a real mathematical risk (there was a 494-dimensional space of
  candidate solutions ignoring the no-negative-ingredients rule). I had a
  *different* solver (linear programming) compute, for every candidate
  ingredient, the range of weight it could possibly carry in any exact
  blend. Result: every range collapsed to a single value — the true one.
  The recipe isn't just recovered; it's provably the only one that fits.
- **Falsifiability + negative controls.** The same battery failed on the
  digit-string data; a "random ingredients" control in the H2 experiment
  confirmed the machinery reports failure on noise.

> **Say this:** "The headline number comes with an oracle gate below it, a
> uniqueness proof behind it, and one recorded failure beside it."

## Step 6 — Vary everything that could matter (the grid)

Each knob isolates one way the method could break:
- **Number of donors K ∈ {2, 5, 10}** — blends of many patients are harder
  (a 10-way blend sits in the "middle" of the crowd).
- **Weight shape** — dominant (0.9 on one donor: tests whether near-twin
  patients get confused) vs spread-out (tests whether faint contributions
  drown).
- **Noise** — clean blends vs blends perturbed at the level of real
  patient-to-blend residuals (tests the resolution floor).
- **Candidate pool** — the true donors handed over (oracle) / the top-50
  most similar by search (how the method really runs) / everyone (control).
- **Two different encoders** — so conclusions don't hinge on one model's
  quirks.

## Step 7 — Results, with each error assigned to its cause

- **The 0.9 came back as 0.884** (error 0.016 against a pass bar of 0.10),
  and the dominant donor was identified **100%** of the time.
- **With the true donors in the pool, recovery is exact and provably
  unique** (the LP certificate above).
- **The only error source is retrieval**: the top-50 similarity search
  finds on average **2.7 of the 5** true donors. Library analogy: the
  answer key is on the shelves, but the catalog search doesn't surface it
  — and no solver can weigh a donor it never sees.
- **Two honest negatives**, both informative: (1) fitting weights on the
  same 50 donors in *concept-code* space vs *text* space gives recipes
  that agree only at chance level — like two librarians cataloging the
  same books by different schemes and disagreeing about which books are
  alike; representation choice changes the causal story. (2) When I asked
  an LLM to *write* the blended patient's note, it copied one donor's note
  nearly verbatim (median 84% overlap) instead of blending — asked to
  merge two recipes, the cook just serves one of them. So generating
  counterfactual *text* needs a trained decoder; that's future work, and
  the experiment proved it with numbers rather than assuming it.

> **Say this:** "Recovery works — 0.9 comes back as 0.88, exactly and
> provably-uniquely when the right donors are present. Everything that
> fails is a replaceable component: the donor search misses donors, fixed
> representations disagree with each other, and an off-the-shelf decoder
> copies instead of blending."

## Step 8 — Why this motivates the proposal (and the satellite move)

Four results, four arrows, one target: the causal core is sound; the
components around it — retrieval, representation, decoder — are the weak
parts, and all three are *learnable*. That is the proposed contribution.
And the mathematics transfers to the satellite domain directly: remote
sensing already runs the same planted-blend validation under the name
**spectral unmixing** (a mixed pixel = a weighted blend of pure material
signatures, weights nonnegative and summing to one) — our plan gives that
machinery causal meaning.

---

## The hard questions, with answers

- **"0.88 out of 0.9 — or exact recovery — isn't that too good? Leakage?"**
  Leakage requires a learned model whose test data contaminated training.
  There is no learned model — a perfect fit *exists by construction*; the
  finding is that it's **unique** (proved by a different solver) and that
  the same test **failed** on bad data. Exactness is normal in noiseless
  inverse problems (compressed sensing publishes exact recovery as
  theorems).
- **"Why should recovering fake blends tell us about real patients?"**
  It's a *necessary* condition, not sufficient — and that's its role. If
  the method can't find weights even when a true recipe exists, its
  weights on real patients are certainly meaningless. It passed the
  necessary test; the sufficiency question (do real-patient weights
  predict real counterfactuals) is the next phase, on data with a
  randomized comparison.
- **"Why these thresholds / this dataset / this encoder?"** Thresholds:
  frozen pre-run in the pre-registration. Dataset: the only openly-licensed
  longitudinal clinical-narrative corpus, verified by reading samples.
  Encoder: chosen by published benchmark evidence, and my own pre-run
  prediction of the winner was wrong — recorded as such, which is what the
  pre-registration is for.
- **"One dataset, 288 patients?"** Stated as the first limitation. The
  battery is cheap (embedding: minutes; solving: seconds per patient) —
  rerunning on a second corpus and more encoders is step one of the paper
  plan.
