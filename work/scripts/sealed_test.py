"""Sealed final-month test (capstone). Frozen recipe from ML-08/ML-09, evaluated ONCE on May -> June 2026.

Train: decide 30 April 2026 (features April, label May).  Test: decide 31 May 2026 (features May, label June).
Nothing about the score, features, or model was changed after this was run.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
sys.path.insert(0, os.path.dirname(__file__))
import lane_data as L  # noqa: E402


def run():
    train = L.add_drop_feature(L.build_frame("2026-04", "2026-05"))
    test = L.add_drop_feature(L.build_frame("2026-05", "2026-06"))
    model = L.make_logistic().fit(train[L.MODEL_FEATURES], train["is_declining"])
    p = model.predict_proba(test[L.MODEL_FEATURES])[:, 1]
    from sklearn.metrics import roc_auc_score
    combined = L.summarize(test, p * (1 + L.drop_per_day(test)))
    rule = L.summarize(test, L.baseline_score(test))
    return {
        "train": {"decision": "2026-04-30", "pages": int(len(train)), "clients": int(train["client_hash_id"].nunique()),
                  "decline_rate": round(float(train["is_declining"].mean()), 4)},
        "test": {"decision": "2026-05-31", "pages": int(len(test)), "clients": int(test["client_hash_id"].nunique()),
                 "decline_rate": round(float(test["is_declining"].mean()), 4)},
        "model_combined": {k: round(v, 4) if isinstance(v, float) else v for k, v in combined.items()},
        "ml07_rule": {k: round(v, 4) if isinstance(v, float) else v for k, v in rule.items()},
        "roc_auc": round(float(roc_auc_score(test["is_declining"], p)), 4),
        "mean_predicted_prob": round(float(p.mean()), 4),
    }


if __name__ == "__main__":
    out = run()
    os.makedirs("work/outputs", exist_ok=True)
    with open("work/outputs/sealed_test_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
