# Complaint Triage

Routing US consumer-finance complaints to a **product category** (9 classes) and an
**issue type** (15 classes), on 100,000 narratives from the CFPB Consumer Complaint
Database.

**Live demo:** *(add your Streamlit app URL)* · **Notebooks:** *(add your Kaggle
profile links)*

---

## Results

Held-out test sets, 15,000 rows per task, opened exactly once.

| Task | Stratified dummy | TF-IDF + LogReg | DistilBERT | Δ | 95% CI |
|---|---|---|---|---|---|
| Product (9 classes) | 0.1101 | 0.7796 | **0.7973** | +0.0178 | [+0.0102, +0.0263] |
| Issue (15 classes) | 0.0660 | 0.5805 | **0.5951** | +0.0147 | [+0.0064, +0.0238] |

Macro-F1. Δ is DistilBERT minus TF-IDF; the interval comes from a paired bootstrap over
1,000 resamples of the test set. P(DistilBERT better) = 100% and 99.9%.

**But the issue headline is misleading, and the error analysis says why — see
[What 0.60 actually measures](#what-060-actually-measures).**

---

## The part worth reading first: the dataset was replaced

The project began on a widely-used Kaggle support-ticket dataset. A schema-verification
and EDA pass, run **before any modelling**, found the labels were independent of the
ticket text. Three diagnostics, any one of which would have been suggestive; together
they are conclusive.

**1. Every class had the same top terms.** Nine of the top twelve TF-IDF terms were
shared by all five categories *and* all four priority levels. Every description was a
template: `"I'm having an issue with the {product_purchased}. Please assist"` plus filler.

**2. Cramér's V was near zero on every field pair.**

| Field pair | Cramér's V |
|---|---|
| Priority × Type | 0.021 |
| Priority × Satisfaction | 0.039 |
| Type × Satisfaction | 0.042 |

Cramér's V rather than a chi-square p-value: at n = 8,469 almost any deviation reaches
significance, so a p-value says nothing about whether the association matters. V
normalises it to an effect size.

**3. The label distributions were uniform.** Priority imbalance ratio **1.06**;
satisfaction ratings 553/549/580/543/544 across the 1–5 scale. Real support queues skew
heavily toward low priority and real CSAT is J-shaped. Flat distributions on every
categorical field at once is the signature of random assignment.

### A gate, not an opinion

To turn that into a decision rather than a judgement call, both datasets were scored by
**the same function at the same sample size** (n = 8,469, the smaller corpus's full
size). The rule: TF-IDF + logistic regression must beat a stratified dummy by ≥ 0.03
macro-F1, or the labels carry no learnable signal.

| Dataset / target | Dummy | TF-IDF | Lift | |
|---|---|---|---|---|
| Original · Ticket Type | 0.1865 | 0.2016 | +0.0151 | FAIL |
| Original · Ticket Priority | 0.2431 | 0.2629 | +0.0198 | FAIL |
| Original · Ticket Subject | 0.0638 | 0.0606 | **−0.0032** | FAIL |
| CFPB · Product | 0.0533 | 0.4877 | **+0.4344** | PASS |
| CFPB · Issue | 0.0600 | 0.5002 | **+0.4403** | PASS |

Ticket Subject scored *below* a stratified coin flip. It was included specifically as the
one alternative target the audit had flagged but not tested, so the switch could not be
called giving up early.

The switch cost three days. Fine-tuning toward 20% accuracy would have cost two weeks and
produced nothing reportable.

---

## Data preparation

CFPB ships 1,282,355 complaints; 383,393 carry a consumer narrative. After cleaning and
deduplication, 349,277 remain, from which 100,000 are sampled per task.

**Taxonomy consolidation.** CFPB renamed its product and issue taxonomy in 2017 and both
vocabularies coexist in the data. `Credit reporting` (31,554 rows) and
`Credit reporting, credit repair services, or other personal consumer reports` (92,338)
are the same class under two names — nothing in the narrative separates them, only the
filing date does. Seven product merges and four issue merges collapse **18 product classes
to 9**. Two classes are dropped rather than merged: `Consumer Loan` split into three
modern classes so has no single correct mapping, and `Other financial service` is a
catch-all. All merges are recorded in `label_maps.json`.

**Redaction stripping.** CFPB masks PII as runs of X — `XX/XX/XXXX` for dates, `XXXX` for
names and account numbers, `{$1,234.00}` for amounts. Measured at **6.6% of all tokens**.
Left in, they would rank among the highest-frequency features in the corpus.

**Deduplicate, then split.** 23,957 narratives are exact duplicates — the same complaint
filed against several companies, or resubmitted. Deduplication runs *before* splitting, so
identical text cannot straddle train and test.

**Per-task splits.** 70/15/15, stratified, seed 42. Product and Issue draw from their own
populations rather than sharing splits — see [Mistakes](#mistakes-and-what-they-cost).

**`max_length = 256`**, chosen by measurement with the actual DistilBERT tokenizer, not by
a words-to-tokens heuristic:

| max_length | Narratives truncated |
|---|---|
| 128 | 53.8% |
| **256** | **25.0%** |
| 384 | 12.6% |
| 512 | 7.3% |

---

## Training

`distilbert-base-uncased`, TensorFlow. Batch 32, AdamW via `create_optimizer` with 10%
linear warmup, class weights from `compute_class_weight("balanced")`, mixed precision.
Best checkpoint selected on **validation macro-F1** by a hand-written callback — Keras has
no macro-F1 metric and `ModelCheckpoint` can only track loss or accuracy, neither of which
is the quantity being optimised on skewed data.

### The three-epoch tie was an artifact of the schedule

At 3 epochs DistilBERT tied TF-IDF on both tasks. That nearly became the project's
conclusion. It was wrong.

`create_optimizer(num_train_steps=...)` decays the learning rate linearly **to zero at the
final step**. A three-epoch run has therefore fully annealed by epoch 3 — it is *finished*,
not merely stopped, and its plateau looks like convergence. Doubling the schedule is a
different experiment, not more of the same:

| Task | 3 epochs | 6 epochs | Gain |
|---|---|---|---|
| Product | 0.7784 | 0.7931 | **+0.0147** |
| Issue | 0.5729 | 0.5875 | **+0.0146** |

Same seed, same data, only the schedule length differs. The 6-epoch run is *behind* at
every matching epoch — its learning rate has not annealed — then overtakes. The two gains
agree to within a thousandth: **the schedule is worth about +0.015 macro-F1 regardless of
task.**

---

## What 0.60 actually measures

DistilBERT scores 0.5951 across 15 issue classes. On its own that reads as a weak
classifier. Reading the rows both models got wrong says otherwise.

**73.6% of DistilBERT's errors never leave the correct semantic family.**

| Family | Classes |
|---|---|
| Credit reporting | Incorrect information · Problem with investigation · Improper use of report |
| Debt collection | Attempts to collect · Written notification · False statements · Took or threatened action · Communication tactics |
| Mortgage & payments | Loan servicing/escrow · Trouble during payment · Struggling to pay · Loan modification · Dealing with servicer |
| Account management | Account opening/closing · Managing an account |

| Granularity | Macro-F1 |
|---|---|
| 15 issue classes | 0.5951 |
| **4 semantic families** | **0.8958** |

The model almost always knows which department a complaint belongs to. What it cannot
reliably do is guess which of several near-synonymous labels the filer chose. **For a
triage system the family is the actionable unit** — you route to a team, not to a
sub-label — so 0.90 describes the capability and 0.60 measures agreement with one person's
choice among overlapping options.

### The labels are genuinely ambiguous

Two test rows scored as DistilBERT errors:

```
TRUE: Took or threatened to take negative or legal action
PRED: Attempts to collect debt not owed
TEXT: "claiming a debt owed when the apartment in question did not
       belong to them ... so i do not owe them anything"

TRUE: Attempts to collect debt not owed
PRED: False statements or representation
TEXT: "deliberately manipulated records ... and misrepresented this
       account in a scheme to blackmail me into paying them"
```

In both, the narrative supports the prediction at least as well as the ground truth. A
third — a debt paid but collection calls continuing — is simultaneously
`Communication tactics` and `Attempts to collect debt not owed`. These are multi-label
problems collapsed into single labels by whoever filed the complaint.

### Both models fail on the same rows

| Task | TF-IDF wrong | DistilBERT wrong | Both wrong | Shared |
|---|---|---|---|---|
| Product | 16.3% | 14.9% | 10.7% | **72%** |
| Issue | 40.2% | 39.3% | 28.7% | **73%** |

A 66M-parameter transformer and a bag-of-words linear model, with almost nothing in common
architecturally, fail on roughly three-quarters of the same rows. The residual error is a
property of the data, not the architecture — which also means ensembling the two would buy
very little.

---

## Methodology notes

**The test set is sealed in code, not by discipline.**

```python
ALLOW_TEST = False   # set True ONLY in notebook 06

def load_split(task, name, base=SPLITS):
    if name == "test" and not ALLOW_TEST:
        raise RuntimeError(f"{task}/test is sealed until notebook 06.")
    return pd.read_csv(os.path.join(base, task, f"{name}.csv"))
```

Every notebook carries this guard set to `False`; notebook 06 is the only place it is
true. Checkpoint selection runs on validation *inside* notebook 06, before the flag is
set, so the test set chose nothing — it only measured.

**Significance is measured two ways, because they answer different questions.** A paired
bootstrap over the test set covers *evaluation* variance: would the margin survive a
different draw of rows? Retraining with a different seed covers *training* variance: would
it survive a rerun. Both are reported; only the first was run systematically.

**A stratified dummy appears in every results table**, so no claim of the form
"X beat the baseline" can conceal "both are near chance."

---

## Mistakes, and what they cost

Four, kept in the record because they were caught by measurement rather than luck.

**Shared splits starved three classes.** Product and Issue originally shared one set of
splits, which required restricting the corpus to the top 15 issues. That deleted 99% of
`Credit card or prepaid card` — 41,660 rows down to ~360 — because credit-card complaints
carry issues ranked 17–24. Three product classes ended up with 15–53 validation rows.

*The signal that exposed it:* accuracy 0.8643 against macro-F1 0.6105. A 25-point gap only
comes from starved classes. Per-task splits took product macro-F1 to 0.7748 — a **+0.164**
improvement from a data-preparation fix, larger than anything architecture contributed.

**A noise floor estimated instead of measured.** Run-to-run variance was estimated by
simulation at 0.0166, and margins under ~0.02 were being treated as ties on that basis.
An interrupted run, restarted at identical settings, later gave a true replication: the
two runs agreed to within **0.002–0.005** at every epoch. The simulation modelled
independent random errors at fixed accuracy, which is far noisier than actually retraining
the same architecture on the same data. Two genuine wins were nearly written up as ties.

**Filter-before-sample, twice.** Rare-class filtering ran before subsampling, so a class
holding 12 rows in a 1M-row frame landed at 1 row in a 25k sample and broke stratified
splitting. The same mistake recurred later with a different threshold.

**A hypothesis that did not survive.** After the 3-epoch runs, the explanation on offer was
that product would gain less from a longer schedule because its classes are separated by
vocabulary while issue's are separated by meaning. It was tidy and it predicted a
difference. The gains came back at +0.0147 and +0.0146. The hypothesis was dropped rather
than softened.

---

## Limitations

- **The family grouping was derived post-hoc** from the confusion structure, which makes
  the 0.8958 partly circular. The rigorous version pre-registers the grouping, or takes it
  from an external taxonomy, then measures. Not done.
- **Single seed per configuration.** The accidental replication suggests run-to-run
  variance is small, but two or three deliberate seeds per task is what would satisfy a
  reviewer.
- **Product had not converged at 6 epochs** — validation macro-F1 was still climbing. A
  win was found; a ceiling was not.
- **Regularisation did not help, inconclusively.** A run with label smoothing, higher
  dropout, lower learning rate and stronger weight decay cut the train/validation gap from
  0.160 to 0.045 but scored 0.5734, below the plain run's 0.5875. It also used 5 epochs
  against 6 and a lower learning rate, so the drop cannot be attributed to the
  regularisers cleanly — a design error in the experiment.
- **Product and Issue train on different row populations**, so their confidence scores are
  not directly comparable.
- **The sentiment indicator is unvalidated.** CFPB carries no sentiment ground truth, so
  the VADER score in the app is descriptive only and labelled as such in the UI.
- **TensorFlow, deliberately, at a cost.** Hugging Face has frozen TF support and removes
  it in `transformers` v5, so this project pins `transformers<5` with `tf-keras`. Keras has
  no macro-F1 metric, and loading the pretrained checkpoint required `use_safetensors=False`
  to route around a broken conversion bridge. Neither happens on the PyTorch path.

---

## Repository

```
notebooks/
  01_verify_and_eda.ipynb        Schema verification, EDA, dataset rejection
  02_signal_gate.ipynb           Comparative gate at matched sample size
  03_splits_and_baselines.ipynb  Consolidation, cleaning, splits, TF-IDF baselines
  04_distilbert_product.ipynb    Product fine-tune, 3 epochs
  05_distilbert_issue.ipynb      Issue fine-tune, 3 epochs
  04b_distilbert_product.ipynb   Product fine-tune, 6 epochs  <- shipped
  05b_distilbert_issue.ipynb     Issue fine-tune, 6 epochs    <- shipped
  05r_distilbert_issue.ipynb     Regularisation attempt (negative result)
  06_final_evaluation.ipynb      Test sets, opened once
app/
  app.py                         Streamlit app
  src/preprocessing.py           clean_text(), identical to the training notebook
  src/predict.py                 predict() and predict_batch()
reports/
  final_results.json             Test metrics and bootstrap intervals
  plots/                         Figures from every notebook
```

### Reproducing

Everything runs on Kaggle Notebooks; no local setup is needed.

1. Import a notebook, set **Internet: ON**. The dataset is pulled from code via
   `kagglehub.dataset_download("selener/consumer-complaint-database")` — no manual
   download.
2. Notebooks 01–03 run on CPU. Notebooks 04b–06 need a **GPU** (~2 h each for the
   fine-tunes, minutes for evaluation).
3. Chain them: commit each notebook with **Save & Run All**, then add its output as an
   input to the next. Notebook 06 needs 03, 04b and 05b.
4. `SEED = 42` throughout.

The 3-epoch notebooks (`04`, `05`) are kept because the schedule finding rests on
comparing them against the 6-epoch runs at a fixed seed. The 6-epoch checkpoints are the
ones notebook 06 selects and the app serves.

---

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
a US federal public record. Narratives are published with consumer consent and with PII
redacted by CFPB. No data is committed to this repository.
