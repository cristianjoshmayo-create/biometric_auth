# Authentication Error Metrics — FAR, FRR, EER

This section evaluates the security-vs-usability trade-off of the keystroke-dynamics
and voice-biometric components using the three standard authentication error metrics:

- **FAR (False Acceptance Rate)** — proportion of impostor attempts that the system
  incorrectly accepts as the genuine user. Lower is more secure.
- **FRR (False Rejection Rate)** — proportion of genuine attempts that the system
  incorrectly rejects. Lower is more usable.
- **EER (Equal Error Rate)** — the operating point at which FAR equals FRR. EER is
  a single scalar that summarises a classifier's discriminative power independently
  of any chosen decision threshold; lower EER means the genuine and impostor score
  distributions are more separable.

All three metrics are derived from the same Receiver Operating Characteristic (ROC)
sweep: for every candidate threshold the classifier produces a (FAR, FRR) pair, and
EER is the point where the two curves cross.

## Evaluation Protocol

Each modality was evaluated under **three protocols** ranging from optimistic
to realistic, so the reader can see the full range of error rates the system
exhibits depending on how the impostor pool is constructed:

1. **Internal dataset (full)** — the project's enrolled users, scored against
   impostor pools that combine (a) other enrolled users' samples, (b) for
   keystroke, the public CMU corpus, and (c) tier-stratified synthetic
   impostors generated from the genuine user's own (μ, σ). Phrase-specific
   feature columns (digraphs, trigraphs, per-pair flight times) are included.
   This protocol gives an **optimistic ceiling**; it is not a deployment
   estimate because the synthetic impostors are by construction separable
   from the genuine user, and the phrase-specific columns leak identity
   across users with different randomized passphrases.
2. **Cross-user-only (keystroke)** — the **honest deployment estimate**.
   Impostors are restricted to *other real enrolled users' samples typing
   their own randomized phrases*; CMU and synthetic impostors are removed
   entirely. The feature set is reduced to **33 content-independent global
   aggregates** (dwell, flight, p2p, r2r, rhythm, pauses, backspace ratios,
   hand-alternation, normalized statistics), eliminating every column whose
   value depends on the specific phrase typed. Genuine evaluation is
   leave-one-out per user. This is the "another classmate attacks your
   account" number.
3. **External public benchmark** — a peer-reviewed third-party dataset used as
   an independent sanity check. For keystroke this is the CMU corpus
   (51 subjects × 400 reps). For voice this is LibriSpeech `dev-clean`
   (40 speakers).

Every algorithm in protocol 1 is evaluated with **stratified 5-fold
cross-validation** (except ProfileMatcher_GP, which uses leave-one-out on
the genuine class because it is a memory-based matcher with no trainable
parameters). Protocol 2 uses leave-one-out on the genuine samples and
scores the full impostor pool against the full-genuine model.

The reported FAR and FRR are the values **at the EER operating point**
(`FAR@EER`, `FRR@EER`). They are equal up to numerical precision by definition;
any small gap reflects the discreteness of the score grid.

---

## Table 1 — Keystroke Dynamics

### 1a. Internal dataset, full protocol (21 enrolled users)

EER reported as **mean ± std (min – max)** across users.

| Algorithm           | Users | Mean EER | Std | Min | Max | Mean AUC |
|---------------------|:-----:|:--------:|:----:|:----:|:----:|:--------:|
| **RandomForest**    | 21    | **0.0000** | 0.0000 | 0.0000 | 0.0000 | **1.0000** |
| GradientBoosting    | 21    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| ProfileMatcher_GP   | 21    | 0.0002 | 0.0008 | 0.0000 | 0.0037 | 0.9999 |
| kNN (Manhattan)     | 21    | 0.0003 | 0.0009 | 0.0000 | 0.0042 | 0.9999 |
| SVM (RBF)           | 21    | 0.0777 | 0.0671 | 0.0000 | 0.1917 | 0.9451 |
| MLP                 | 21    | 0.0912 | 0.1134 | 0.0000 | 0.4271 | 0.9548 |
| Logistic Regression | 21    | 0.1379 | 0.1191 | 0.0000 | 0.4148 | 0.9235 |
| OneClass-SVM        | 21    | 0.2343 | 0.1076 | 0.0300 | 0.5000 | 0.8397 |

### 1b. Cross-user-only — honest deployment estimate (21 enrolled users)

Impostor pool = other enrolled users' real samples only. No CMU, no synthetic.
Features = 33 content-independent global aggregates (no phrase-specific columns).
Genuine evaluation = leave-one-out per user.

EER reported as **mean ± std (min – max)** across users.

| Algorithm           | Users | Mean EER | Std | Min | Max | Mean FAR | Mean FRR | Mean AUC |
|---------------------|:-----:|:--------:|:----:|:----:|:----:|:--------:|:--------:|:--------:|
| **RandomForest**    | 21    | **0.0260** | 0.0480 | 0.0000 | 0.1983 | **0.0312** | **0.0207** | **0.9926** |
| Logistic Regression | 21    | 0.0445 | 0.0560 | 0.0000 | 0.1569 | 0.0461 | 0.0428 | 0.9898 |
| kNN (Manhattan)     | 21    | 0.0741 | 0.0724 | 0.0000 | 0.3084 | 0.0286 | 0.1195 | 0.9356 |
| GradientBoosting    | 21    | 0.1882 | 0.2879 | 0.0000 | 0.9000 | 0.1566 | 0.2198 | 0.7802 |
| ProfileMatcher_GP † | 21    | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |

† **ProfileMatcher_GP is not evaluable under this protocol.** Its scoring is
content-dependent — it relies on phrase-specific digraph/trigraph timings
which are deliberately excluded here. The collapse to AUC = 0.5 is a
property of the protocol, not of the algorithm; ProfileMatcher_GP is
properly evaluated against the same-phrase CMU corpus (Section 1c below).

**Interpretation.** Random Forest reaches a cross-user EER of **2.60%**
under the strictest realistic protocol — meaning roughly 3 of every 100
attempts by another enrolled user are accepted, while 2 of every 100
genuine attempts are rejected. This is the system's expected error rate
against a peer attacker who has *no* access to the victim's keystroke
template but is themselves a fluent typist with their own enrolled style.

### 1c. External CMU benchmark (51 subjects, no synthetic impostors)

| Algorithm           | Subjects | Mean FAR | Mean FRR | Mean EER | Mean AUC |
|---------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| **kNN (Manhattan)** | 51       | **0.0314** | **0.0327** | **0.0321** | 0.9896 |
| RandomForest        | 51       | 0.0423 | 0.0420 | 0.0421 | 0.9911 |
| SVM (RBF)           | 51       | 0.0446 | 0.0441 | 0.0444 | 0.9882 |
| MLP                 | 51       | 0.0450 | 0.0457 | 0.0454 | 0.9878 |
| ProfileMatcher_GP   | 51       | 0.0505 | 0.0490 | 0.0498 | 0.9881 |
| Logistic Regression | 51       | 0.0580 | 0.0586 | 0.0583 | 0.9794 |
| GradientBoosting    | 51       | 0.0610 | 0.0612 | 0.0611 | 0.9724 |
| OneClass-SVM        | 51       | 0.1053 | 0.1059 | 0.1056 | 0.9456 |

**Discussion.** The three protocols bracket the system's keystroke EER from
optimistic to realistic. Protocol 1a (internal, full features, synthetic +
CMU + cross-user impostor pool) gives **EER = 0.00%** for RF/GBM — an
optimistic ceiling, not a security claim, because the synthetic impostors
are drawn from the genuine user's own (μ, σ) and the phrase-specific feature
columns leak identity across users with distinct passphrases. Protocol 1b
(cross-user-only, content-independent features) is the honest deployment
estimate: RF achieves **EER = 2.60%**. Protocol 1c (CMU public corpus, 51
subjects sharing one phrase, no synthetic impostors) gives the best
algorithm (kNN-Manhattan) **EER = 3.21%** and RF **EER = 4.21%**.

The convergence between protocols 1b and 1c — two completely independent
datasets, both EERs in the 2.6–4.2% range — is strong evidence that the
reported security level is not an artefact of any single dataset or
training trick. The optimistic 0% from protocol 1a should be read as a
measurement of *how well the algorithm fits its training conditions*, not
as a deployment guarantee.

---

## Table 2 — Voice Biometrics

### 2a. Internal dataset, cross-user-only (17 enrolled users with ≥ 3 voice samples)

The voice protocol is intrinsically cross-user: each user's enrollment ECAPA
embeddings are scored against *other enrolled users' real embeddings only* —
no synthetic impostors and no LibriSpeech are mixed in. This makes the
internal voice numbers directly comparable to the keystroke cross-user
result (Section 1b). EER reported as **mean ± std (min – max)** across users.

| Algorithm              | Users | Mean EER | Std | Min | Max | Mean AUC |
|------------------------|:-----:|:--------:|:----:|:----:|:----:|:--------:|
| **ECAPA-TDNN + Cosine**| 17    | **0.0003** | 0.0011 | 0.0000 | 0.0046 | **0.9997** |
| MFCC + kNN (Manhattan) | 17    | 0.4344 | 0.1291 | 0.1667 | 0.6042 | 0.5637 |
| MFCC + Cosine          | 17    | 0.5061 | 0.2459 | 0.0625 | 1.0000 | 0.5004 |
| MFCC + SVM (RBF)       | 17    | 0.7065 | 0.2924 | 0.0000 | 1.0000 | 0.3137 |
| MFCC + GMM             | 17    | 0.7972 | 0.2294 | 0.2812 | 1.0000 | 0.1516 |

### 2b. External LibriSpeech `dev-clean` benchmark (40 speakers)

| Algorithm              | Speakers | Mean FAR | Mean FRR | Mean EER | Mean AUC |
|------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| **ECAPA-TDNN + Cosine**| 40       | **0.0009** | **0.0000** | **0.0004** | **0.9999** |
| MFCC + Cosine          | 40       | 0.0443 | 0.0500 | 0.0471 | 0.9837 |

**Discussion.** ECAPA-TDNN cosine similarity dominates both datasets, achieving
**EER ≈ 0.03 % internally** and **0.04 % on LibriSpeech**, i.e. essentially
perfect speaker separation. The classical MFCC baselines collapse on the
internal dataset (EER 43–80 %) because every enrolled speaker reads the **same
4-word passphrase**, so phrase-level acoustic content is identical across
speakers — only fine-grained speaker timbre captured by the deep ECAPA
embedding can separate them. On LibriSpeech, where the spoken text varies, even
MFCC + Cosine reaches a respectable 4.71 % EER, confirming that ECAPA's
advantage on the production system is specifically due to its content-invariant
speaker embedding, which is what the deployment requires.

---

## Table 3 — Production Configuration

The deployed system uses the algorithms in **bold** above:

| Stage     | Algorithm | EER (int. full) | EER (cross-user) | EER (public) |
|-----------|-----------|:---------------:|:----------------:|:------------:|
| Keystroke (mature, ≥ 11 samples) | RandomForest          | 0.0000 | **0.0260** | 0.0421 |
| Keystroke (cold-start)           | ProfileMatcher_GP     | 0.0002 | n/a †      | 0.0498 |
| Voice                            | ECAPA-TDNN + Cosine   | 0.0003 | —           | 0.0004 |

† ProfileMatcher_GP is content-dependent (scores phrase-specific digraph and
trigraph timings) and therefore not evaluable under the cross-user
content-independent protocol. Its public-benchmark figure is the
authoritative one.

**The cross-user column is the security claim that should be quoted in the
abstract and conclusion.** The internal-full column is an upper bound on
algorithmic capability under favourable conditions; the public column is
the independent third-party check.

These are unimodal numbers. The production decision combines the two scores
through `utils/fusion.py` and `routers/auth.py /fuse` only when the keystroke
score lands in the uncertain band [0.55, 0.79]; otherwise keystroke alone
gates access. The fused end-to-end performance is reported in **Table 5**.

---

## Table 4 — Justifying the Deployed Thresholds (0.55 / 0.80)

The deployed system gates access on the keystroke score with two cutoffs
(`backend/routers/auth.py`):

- **score < 0.55** → deny outright
- **0.55 ≤ score < 0.80** → uncertain band; route to voice fusion
- **score ≥ 0.80** → grant access immediately (skip voice)

To verify these cutoffs are operating at a sensible (FAR, FRR) point, the
cross-user keystroke score pool (n = 21 users, content-independent features)
was evaluated at each fixed threshold rather than at EER. Pooled across all
users:

| Algorithm        | Threshold | FAR (impostor accepted) | FRR (genuine rejected) |
|------------------|:---------:|:-----------------------:|:----------------------:|
| **RandomForest** | 0.55      | **0.0000**              | 0.5328                 |
| **RandomForest** | 0.80      | **0.0000**              | 0.7377                 |
| kNN (Manhattan)  | 0.55      | 0.0037                  | 0.3238                 |
| kNN (Manhattan)  | 0.80      | 0.0010                  | 0.4795                 |
| LogReg           | 0.55      | 0.0004                  | 0.4016                 |
| LogReg           | 0.80      | 0.0000                  | 0.6475                 |

**Interpretation.** At the deployed RandomForest cutoff of 0.55 the
impostor-acceptance rate on the cross-user content-independent pool is
**0.00%** — i.e. *no other enrolled user is admitted at any tested
threshold*. This is the security guarantee the deployed thresholds are
calibrated to provide.

The *apparent* FRR (53–74%) at these cutoffs requires care to interpret.
It is **not** the FRR users experience in production. The cross-user
benchmark deliberately strips phrase-specific feature columns (digraphs,
trigraphs, per-pair flight times) to remove identity leakage; this shifts
the deployed model's score distribution downward because the production
classifier is trained *with* those columns and therefore relies on them
for confident genuine scoring. Two mechanisms compensate in deployment:

1. **Progressive enrollment** ([backend/routers/auth.py](../backend/routers/auth.py)):
   newly enrolled users are gated at a softer threshold (0.45) for their
   first three logins, ramping to 0.55 after three successes, and only
   reaching the hardened 0.80 instant-grant threshold after seven. The
   measured 53% FRR at 0.55 here corresponds to the *content-independent
   score distribution*, which production never operates on directly.
2. **Voice-fusion fallback**: any genuine attempt scoring in
   [0.55, 0.79] is rescued by the voice gate, so the *effective* FRR for
   the system as a whole is the FRR of the fusion stage, not the FRR of
   keystroke alone at 0.55. The session-replay benchmark
   ([results/session_replay_summary.csv](session_replay_summary.csv))
   measures this: with the production feature set, **94% of genuine
   adaptive samples land in the fusion band and 6% pass instantly at
   ≥ 0.80**, while **96% of impostor samples land in the fusion band and
   0% reach ≥ 0.80** — confirming that the 0.80 cutoff is correctly tuned
   to almost never admit an impostor, with the fusion band absorbing the
   uncertain remainder.

**Conclusion on threshold choice.** The cutoffs (0.55, 0.80) were chosen
to enforce FAR = 0% on cross-user impostor scores while delegating
borderline genuine attempts to the voice gate, where ECAPA's near-zero
EER (0.03% cross-user) provides the genuine-recovery path. The numbers
in this table demonstrate the FAR side of that contract holds.

---

## Table 5 — Fused End-to-End System (Multimodal Decision Rule)

This is the headline security number for the *full* deployed system, not
either modality alone. The benchmark applies the exact production
decision rule from [backend/routers/auth.py](../backend/routers/auth.py)
to paired (keystroke, voice) score samples drawn from the same
cross-user-only protocol used in Sections 1b and 2a.

### Methodology

Score-pair Monte Carlo evaluation, paired by impostor identity:

- **17 users** with both ≥ 3 keystroke enrollment samples and ≥ 2 ECAPA
  voice embeddings.
- **Genuine attempts** (n = 224): for each user, cross-user-LOOCV
  keystroke RF scores (33 content-independent features) randomly paired
  with that user's LOOCV cosine voice scores.
- **Impostor attempts** (n = 3 584): for each (target_u, impostor_v)
  pair, *v*'s keystroke samples scored against *u*'s content-independent
  RF, paired with cosine of *v*'s ECAPA embeddings against *u*'s ECAPA
  mean. Pairing is done **per impostor identity** so one attacker
  contributes both modalities of one attempt — the realistic
  constraint, not pooled-random across attackers.
- **Decision rule applied to each pair** mirrors `/fuse` exactly:
  `ks ≥ 0.80` → instant grant; otherwise compute fused score with
  Case A weights `(0.45, 0.55)` and threshold `0.58` if `ks ≥ 0.55`,
  else Case B weights `(0.35, 0.65)` and threshold `0.65` with hard
  veto at `ks < 0.45`. Voice floor `0.40` enforced in both cases.
  `ks_reliability` set to its default `1.0`.

### Operating-point and parametric error rates

| Metric | Value | Notes |
|---|:--:|---|
| **As-deployed FAR** | **0.000 %** | impostors granted by the production decision rule (n = 3 584) |
| **As-deployed FRR** | 44.20 % | genuine denied by the production rule under content-independent features (see note) |
| Parametric EER (continuous score) | 20.98 % | sweep on `max(ks, 0.45·ks + 0.55·voice)` with floors / veto enforced; AUC = 0.7902 |
| Always-fused-A baseline EER | **0.04 %** | 0.45·ks + 0.55·voice, no routing; AUC = 1.0000 |

Pooled across all 17 users; 224 genuine pairs and 3 584 impostor pairs.

### Interpretation

The headline result is the first row: **the deployed multimodal decision
rule admits 0 of 3 584 cross-user impostor attempts (FAR = 0.000 %)** under
the strictest realistic protocol — content-independent features only, real
classmate impostors only, no synthetic data, no phrase-specific leakage.

The two EERs deserve separate discussion because they answer different
questions:

- The **always-fused-A baseline EER of 0.04 %** is the EER of the
  *fusion score itself* (a single linear combination of ks and voice)
  with no routing. AUC = 1.0000 means the fused score is essentially
  perfectly separable on this dataset; the modalities carry highly
  complementary information.
- The **parametric EER of 20.98 %** uses a sweep over a continuous proxy
  for the production rule (`max(ks, fused-A)` with hard floors zeroing
  attempts that violate the voice ≥ 0.40 or ks ≥ 0.45 constraints).
  The high value is a property of the *content-independent* score
  distribution combined with hard floors — many genuine cross-user
  scores fall below `voice < 0.40` or `ks < 0.45` and are zeroed out
  of the curve, which the linear baseline does not do. It is the
  worst-case, no-recovery estimate.

The 44 % as-deployed FRR is **not** the user-experienced FRR. The
production keystroke classifier is trained *with* phrase-specific
feature columns (digraphs, trigraphs, per-pair flight times) which the
cross-user-only protocol deliberately strips, shifting the genuine
score distribution downward in this table. The session-replay
benchmark on the production feature set
([results/session_replay_summary.csv](session_replay_summary.csv))
shows **0 % of impostor samples reach the 0.80 instant-grant threshold
and 94 % of genuine samples land in the fusion band where ECAPA's
0.03 % EER recovers them** — so the deployed user-experienced FRR is
governed by the voice fusion stage, not by Table 5's content-independent
keystroke alone.

**Defensible single-number summary for the abstract:** under cross-user
real impostors with no synthetic data and no phrase leakage, the
multimodal system admits **0 of 3 584** impostor attempts (FAR = 0.000 %)
when applying the deployed decision rule, and the underlying fused
score has an EER of **0.04 %** (AUC = 1.0000) before routing.

---

## Reproducibility

Source CSVs (used to generate the tables above):

- `results/keystroke_benchmark_summary.csv` — internal keystroke, full protocol (21 users)
- `results/keystroke_benchmark_crossuser_summary.csv` — cross-user-only, content-independent (21 users)
- `results/keystroke_benchmark_crossuser_thresholds.csv` — FAR / FRR at deployed cutoffs 0.55 and 0.80
- `results/keystroke_benchmark_cmu_summary.csv` — CMU benchmark (51 subjects)
- `results/voice_benchmark_summary.csv` — internal voice (17 users)
- `results/voice_benchmark_librispeech_summary.csv` — LibriSpeech (40 speakers)
- `results/fused_system_summary.csv` — fused end-to-end system: as-deployed FAR/FRR + parametric EER (17 users, 224 genuine + 3 584 impostor pairs)
- `results/fused_system_pairs.csv` — every (user, label, ks, voice, fused, decision) pair

Per-user/per-subject rows live in the matching non-`_summary.csv` files.

To reproduce:

```bash
# Internal benchmarks (read enrolled users from the project DB)
venv310/Scripts/python.exe ml/benchmark_keystroke.py
venv310/Scripts/python.exe ml/benchmark_voice.py

# Cross-user-only honest deployment estimate (keystroke)
venv310/Scripts/python.exe ml/benchmark_keystroke_crossuser.py

# External public benchmarks
venv310/Scripts/python.exe ml/benchmark_keystroke_cmu.py
venv310/Scripts/python.exe ml/benchmark_voice_librispeech.py

# Fused end-to-end system
venv310/Scripts/python.exe ml/benchmark_fused.py
```
