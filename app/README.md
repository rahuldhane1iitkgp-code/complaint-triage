# Streamlit app

Deployed on Streamlit Community Cloud from this repository.

- **App file:** `app/app.py`
- **Dependencies:** `requirements.txt` at the repository root
- **Models:** `app/models/tfidf_lr_product.joblib` and `app/models/tfidf_lr_issue.joblib`

## Before it will run

The two joblib pipelines are produced by `notebooks/03_splits_and_baselines.ipynb` and
must be committed into `app/models/`. Without them the app stops at startup with a clear
error naming the missing file.

Pin `scikit-learn` in the root `requirements.txt` to the version that fitted them —
joblib pickles are not guaranteed to load across scikit-learn versions. Print it in the
notebook with `import sklearn; print(sklearn.__version__)`.

## Local run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
