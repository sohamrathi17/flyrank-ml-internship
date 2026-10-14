"""Shared data + metrics for my Refresh / Content Opportunity lane (ML-08 onward).

One row = one page at a decision moment (the last day of the feature month).
Features come from the feature month only; the label comes from the NEXT month:
    is_declining = label-month avg daily impressions < 0.8 x feature-month avg daily impressions
Slice: GSC data on every day of both months (gsc_data_available IS TRUE) and >= 100 impressions
in the feature month. Same definition as work/notebooks/w03_data_contract.ipynb (ML-04).

Parquet files are downloaded once into the Hugging Face cache (outside the repo, never committed).
"""
import os

import duckdb
import numpy as np
import pandas as pd

REPO_ID = "FlyRank/internship-warehouse"
VOLUME_FLOOR = 100
DECLINE_CUT = 0.8  # label: next-month daily impressions below 80% of this month's

# Numeric features, all knowable on the last day of the feature month
NUMERIC_FEATURES = [
    # the five from ML-04
    "log_imp", "avg_position", "ctr", "age_days", "momentum",
    # extra feature-month signals
    "log_clicks", "days_with_imp", "imp_cv", "position_change", "top_day_share",
    # keyword metadata from dim_content (set when the keyword was chosen, not when the page was edited)
    "log_search_volume", "has_search_volume", "competition", "keyword_token_count",
]
CATEGORICAL_FEATURES = ["content_type", "main_intent"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Never features: label window, label itself, IDs, snapshot dates that can postdate the decision moment
NEVER_FEATURES = ["imp_next", "is_declining", "impressions_lost", "client_hash_id", "content_hash_id",
                  "content_updated_date", "last_optimized_date"]


def get_token():
    """Token order: env var -> Colab Secret -> local `hf auth login` cache -> prompt."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if not token:
        from huggingface_hub import get_token as cached_token
        token = cached_token()
    if not token:
        import getpass
        token = getpass.getpass("Paste your Hugging Face READ token (hf_...): ")
    return token


def hf_file(path, token):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, path, repo_type="dataset", token=token)


def month_path(month, token):
    return hf_file(f"fact_content_daily_performance/month={month}/data_0.parquet", token)


def build_frame(feature_month, label_month, token=None):
    """One row per page: feature-month signals + next-month label. Months as 'YYYY-MM'."""
    token = token or get_token()
    feat = f"read_parquet('{month_path(feature_month, token)}')"
    lab = f"read_parquet('{month_path(label_month, token)}')"
    dim = f"read_parquet('{hf_file('dim_content.parquet', token)}')"

    start = pd.Timestamp(f"{feature_month}-01")
    decision = start + pd.offsets.MonthEnd(0)
    feat_days = decision.day
    label_days = (pd.Timestamp(f"{label_month}-01") + pd.offsets.MonthEnd(0)).day
    early_days = (feat_days + 1) // 2                        # March: days 1-16 early, 17-31 late (as in ML-04)
    late_days = feat_days - early_days
    mid = start + pd.Timedelta(days=early_days)              # second half starts here

    con = duckdb.connect()
    frame = con.sql(f"""
        WITH f AS (
            SELECT client_hash_id, content_hash_id,
                   SUM(gsc_impressions)                                   AS imp,
                   SUM(gsc_clicks)                                        AS clicks,
                   COUNT(*) FILTER (WHERE gsc_impressions > 0)            AS days_with_imp,
                   STDDEV_POP(gsc_impressions) / NULLIF(AVG(gsc_impressions), 0) AS imp_cv,
                   MAX(gsc_impressions)::DOUBLE / NULLIF(SUM(gsc_impressions), 0) AS top_day_share,
                   SUM(CASE WHEN gsc_avg_position > 0 THEN gsc_sum_position END)::DOUBLE
                     / NULLIF(SUM(CASE WHEN gsc_avg_position > 0 THEN gsc_impressions END), 0) AS avg_position,
                   SUM(CASE WHEN gsc_avg_position > 0 AND report_date >= DATE '{mid.date()}' THEN gsc_sum_position END)::DOUBLE
                     / NULLIF(SUM(CASE WHEN gsc_avg_position > 0 AND report_date >= DATE '{mid.date()}' THEN gsc_impressions END), 0) AS pos_late,
                   SUM(CASE WHEN gsc_avg_position > 0 AND report_date <  DATE '{mid.date()}' THEN gsc_sum_position END)::DOUBLE
                     / NULLIF(SUM(CASE WHEN gsc_avg_position > 0 AND report_date <  DATE '{mid.date()}' THEN gsc_impressions END), 0) AS pos_early,
                   SUM(CASE WHEN report_date >= DATE '{mid.date()}' THEN gsc_impressions END) / {late_days}.0  AS imp_late_per_day,
                   SUM(CASE WHEN report_date <  DATE '{mid.date()}' THEN gsc_impressions END) / {early_days}.0 AS imp_early_per_day
            FROM {feat}
            WHERE gsc_data_available IS TRUE
            GROUP BY ALL
            HAVING COUNT(*) = {feat_days} AND SUM(gsc_impressions) >= {VOLUME_FLOOR}
        ),
        l AS (   -- label window: used for the label only
            SELECT client_hash_id, content_hash_id, SUM(gsc_impressions) AS imp_next
            FROM {lab}
            WHERE gsc_data_available IS TRUE
            GROUP BY ALL
            HAVING COUNT(*) = {label_days}
        )
        SELECT f.*, l.imp_next,
               d.content_created_date, d.content_type, d.main_intent,
               d.search_volume, d.competition, d.keyword_token_count
        FROM f
        JOIN l USING (client_hash_id, content_hash_id)
        LEFT JOIN {dim} d USING (client_hash_id, content_hash_id)
    """).df()

    frame["decision_date"] = decision
    frame["log_imp"] = np.log1p(frame["imp"])
    frame["log_clicks"] = np.log1p(frame["clicks"])
    frame["ctr"] = frame["clicks"] / frame["imp"]
    frame["age_days"] = (decision - pd.to_datetime(frame["content_created_date"])).dt.days
    frame["momentum"] = (frame["imp_late_per_day"] + 1) / (frame["imp_early_per_day"] + 1)
    frame["position_change"] = frame["pos_late"] - frame["pos_early"]   # > 0 means slipping down
    frame["has_search_volume"] = frame["search_volume"].notna().astype(int)
    frame["log_search_volume"] = np.log1p(frame["search_volume"])
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].fillna("unknown").astype("category")

    # Label + impact (evaluation only)
    frame["is_declining"] = ((frame["imp_next"] / label_days) < DECLINE_CUT * (frame["imp"] / feat_days)).astype(int)
    frame["impressions_lost"] = (frame["imp"] / feat_days * label_days - frame["imp_next"]).clip(lower=0)

    assert frame["content_hash_id"].is_unique, "grain broke: more than one row per page"
    assert (frame["age_days"].dropna() >= 0).all(), "a page was created after the decision moment"
    # DuckDB does not guarantee row order; a fixed order makes folds and models reproducible
    return frame.sort_values(["client_hash_id", "content_hash_id"]).reset_index(drop=True)


def per_client_metrics(frame, score, k=20, min_pages=20):
    """Macro precision@k and share of lost impressions captured in the top k, per client."""
    d = frame.assign(_score=np.asarray(score, dtype=float))
    rows = []
    for client, g in d.groupby("client_hash_id", observed=True):
        if len(g) < min_pages:
            continue
        top = g.sort_values(["_score", "content_hash_id"], ascending=[False, True]).head(k)
        rows.append({
            "client": client,
            "pages": len(g),
            "base_rate": g["is_declining"].mean(),
            "precision_at_k": top["is_declining"].mean(),
            "loss_captured_at_k": top["impressions_lost"].sum() / max(g["impressions_lost"].sum(), 1e-9),
        })
    return pd.DataFrame(rows)


def summarize(frame, score, k=20, min_pages=20):
    m = per_client_metrics(frame, score, k, min_pages)
    return {
        "clients": int(len(m)),
        f"precision@{k}": float(m["precision_at_k"].mean()),
        f"loss_captured@{k}": float(m["loss_captured_at_k"].mean()),
        "base_rate": float(m["base_rate"].mean()),
    }


# ---------------------------------------------------------------------------
# The ML-07 baseline rule (frozen), same logic as work/notebooks/w04_baseline_score.ipynb
TIER_EDGES = [0, 3, 10, 20, 50, np.inf]
TIER_LABELS = ["top_3", "page_1", "striking", "page_3_5", "deep"]


def baseline_score(frame):
    """Score = max(0, early-month imp/day - late-month imp/day) x 1.5 if CTR is below its position-tier median.

    Returns the score with the ML-07 tie-break (bigger month impressions first) folded in, so
    sorting by it reproduces the ML-07 queue order.
    """
    tier = pd.cut(frame["avg_position"], TIER_EDGES, labels=TIER_LABELS)
    tier_median_ctr = frame.groupby(tier, observed=True)["ctr"].median()
    ctr_vs_tier = frame["ctr"] / tier.map(tier_median_ctr).astype(float)
    low_ctr = (frame["clicks"] > 0) & (ctr_vs_tier < 1)
    drop = (frame["imp_early_per_day"] - frame["imp_late_per_day"]).clip(lower=0)
    score = (drop * np.where(low_ctr, 1.5, 1.0)).round(2)
    return score + frame["imp"].rank(pct=True) * 1e-3   # tie-break only, < 0.01


# ---------------------------------------------------------------------------
# Models (ML-08 onward). Seeds fixed at 42.
SEED = 42


def add_drop_feature(frame):
    """log of the within-month daily impression drop (early half minus late half, floored at 0)."""
    frame["log_drop"] = np.log1p((frame["imp_early_per_day"] - frame["imp_late_per_day"]).clip(lower=0))
    return frame


MODEL_NUMERIC = NUMERIC_FEATURES + ["log_drop"]
MODEL_FEATURES = MODEL_NUMERIC + CATEGORICAL_FEATURES


def make_logistic():
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), MODEL_NUMERIC),
            (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)),
        LogisticRegression(max_iter=2000))


def make_boosted_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=SEED, categorical_features="from_dtype",
                                          max_iter=300, learning_rate=0.05)


def make_boosted_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=SEED, categorical_features="from_dtype",
                                         max_iter=300, learning_rate=0.05)


def drop_per_day(frame):
    return (frame["imp_early_per_day"] - frame["imp_late_per_day"]).clip(lower=0)


def out_of_fold(frame, make_model, target, proba=True, n_splits=5):
    """Grouped K-fold by client: every page is scored by a model that never saw its client."""
    from sklearn.model_selection import GroupKFold
    X, y = frame[MODEL_FEATURES], frame[target]
    preds = np.zeros(len(frame))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, frame["client_hash_id"]):
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = (model.predict_proba(X.iloc[test_idx])[:, 1] if proba
                           else model.predict(X.iloc[test_idx]))
    return preds
