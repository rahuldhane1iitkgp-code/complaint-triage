import io
import os

import pandas as pd
import streamlit as st

from src.predict import predict, predict_batch, load_models, TASKS

st.set_page_config(page_title="Complaint Triage", page_icon="🔍", layout="wide")

# Held-out test figures, filled in from notebook 06's final_results.json.
RESULTS = {
    "product": {"classes": 9, "macro_f1": None},
    "issue": {"classes": 15, "macro_f1": None},
}

EXAMPLE = (
    "I have disputed this account with all three bureaus multiple times. "
    "The collector keeps reporting a balance of {$2400.00} that I never owed, "
    "and on XX/XX/XXXX they called my workplace twice after I asked them in "
    "writing to stop contacting me there. I have sent proof of payment and "
    "nothing has been corrected on my credit report."
)


@st.cache_resource(show_spinner="Loading models…")
def warm():
    load_models()
    return True


def confidence_row(result, task, title):
    st.markdown(f"**{title}**")
    st.markdown(f"### {result[task]['label']}")
    st.progress(result[task]["confidence"])
    st.caption(f"confidence {result[task]['confidence']:.1%}")
    with st.expander("Alternatives considered"):
        for label, p in result[task]["top_k"][1:]:
            st.write(f"{p:.1%} — {label}")


st.title("Complaint Triage")
st.caption(
    "Routes a US consumer-finance complaint to a product category and an issue type. "
    "Trained on the CFPB Consumer Complaint Database."
)

try:
    warm()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single complaint", "Batch upload", "About the models"])

# ------------------------------------------------------------------ single
with tab1:
    text = st.text_area(
        "Complaint narrative",
        value=EXAMPLE,
        height=200,
        help="Paste the customer's own words. CFPB-style XXXX redactions are fine.",
    )
    if st.button("Classify", type="primary"):
        result = predict(text)
        if not result["ok"]:
            st.warning(result["reason"])
        else:
            left, right = st.columns(2)
            with left:
                confidence_row(result, "product", "Product")
            with right:
                confidence_row(result, "issue", "Issue")

            st.divider()
            tone = result["tone"]
            c1, c2 = st.columns([1, 3])
            c1.metric("Tone", tone["label"], f"{tone['compound']:+.2f}")
            c2.info(
                "**Tone is not a validated prediction.** It is a VADER lexicon score "
                "over the raw text, shown as a descriptive signal. The CFPB database "
                "carries no sentiment label, so this was never evaluated against "
                "ground truth — unlike the two classifiers above."
            )
            st.caption(f"{result['n_words']} words after cleaning")

# ------------------------------------------------------------------ batch
with tab2:
    st.markdown(
        "Upload a CSV with a text column. Every row is classified and returned "
        "with confidences."
    )
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        col = st.selectbox(
            "Which column holds the complaint text?",
            df.columns,
            index=max(
                0,
                next((i for i, c in enumerate(df.columns)
                      if "text" in c.lower() or "narrative" in c.lower()
                      or "complaint" in c.lower()), 0),
            ),
        )
        if st.button(f"Classify {len(df):,} rows", type="primary"):
            with st.spinner("Classifying…"):
                out = pd.DataFrame(predict_batch(df[col]))
            result = pd.concat([df.reset_index(drop=True), out], axis=1)
            st.session_state["batch"] = result

    if "batch" in st.session_state:
        result = st.session_state["batch"]
        skipped = int((~result["classified"]).sum())
        if skipped:
            st.warning(f"{skipped} row(s) too short to classify after cleaning.")
        st.dataframe(result.head(50), use_container_width=True)
        buf = io.StringIO()
        result.to_csv(buf, index=False)
        st.download_button(
            "Download predictions CSV",
            buf.getvalue(),
            file_name="triage_predictions.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Distribution across this batch")
        c1, c2, c3 = st.columns(3)
        done = result[result["classified"]]
        with c1:
            st.markdown("**Product**")
            st.bar_chart(done["product"].value_counts())
        with c2:
            st.markdown("**Issue**")
            st.bar_chart(done["issue"].value_counts())
        with c3:
            st.markdown("**Tone** *(unvalidated)*")
            st.bar_chart(result["tone"].value_counts())

        low = done[done["product_confidence"] < 0.5]
        st.caption(
            f"{len(low)} of {len(done)} rows ({len(low)/max(len(done),1):.0%}) fall "
            "below 50% product confidence — the queue a human should review first."
        )
    else:
        st.info("No batch loaded yet. Upload a CSV above to see distributions here.")

# ------------------------------------------------------------------ about
with tab3:
    st.subheader("What is serving these predictions")
    st.markdown(
        """
**TF-IDF (1–2 grams, 50k features) + Logistic Regression**, class-weighted,
fit on 70,000 CFPB complaint narratives per task.

### DistilBERT is more accurate, and is not what this app serves

On the held-out test sets a fine-tuned DistilBERT beat these models:

| Task | TF-IDF | DistilBERT | Difference |
|---|---|---|---|
| Product | 0.7796 | 0.7973 | +0.0178 |
| Issue | 0.5805 | 0.5951 | +0.0147 |

Macro-F1. Both differences are real — paired-bootstrap intervals exclude zero.

The linear model is served anyway, and the trade is worth stating plainly: it runs
on a free CPU tier with no cold start and no 500MB of weights, for about 1.8
points of macro-F1. The DistilBERT checkpoints and the full comparison are in the
repository. **This is a hosting constraint, not a claim that the models are
equivalent.**
"""
    )
    st.subheader("Known limitations")
    st.markdown(
        """
- **Issue labels overlap.** *False statements or representation*, *Took or
  threatened to take negative or legal action* and *Attempts to collect debt not
  owed* describe overlapping complaints. The person filing picks one; a model
  reading only the narrative cannot recover that choice. Both models fail on the
  same classes, which points at label ambiguity rather than model weakness.
- **Product and issue were trained on different row populations**, so their
  confidences are not directly comparable to one another.
- **Tone is unvalidated** and should not be used for routing.
- Trained on US consumer-finance complaints. Behaviour on other domains is
  untested.
"""
    )
