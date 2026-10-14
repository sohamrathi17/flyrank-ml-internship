# Capstone Report: Refresh / Content Opportunity Scoring

- **Author:** Soham Rathi
- **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
- **Repo:** https://github.com/sohamrathi17/flyrank-ml-internship
- **Paper:** https://sohamrathi17.github.io/flyrank-ml-internship/
- **Date:** October 2026

## 0. Abstract

Content editors can review only a handful of a site's pages each month, so I asked which pages should top that list to catch the ones that will keep losing search visibility next month. Using the FlyRank ML Internship warehouse (about 79 million rows of pseudonymised daily Search Console data), I built monthly snapshots with features from one month and the outcome (daily impressions falling more than 20%) from the next. I compared a transparent rule ("how much has this page already slid within the month?") with logistic regression and gradient-boosted trees on 16 signals, using client-grouped cross-validation, a time-aware split, a sealed final month evaluated once, and a client-level bootstrap. The rule put about 71-72% true decliners in each client's top 20 against a base rate near 49% and held that level on the sealed month; the best learned score was about one point higher, within bootstrap noise. The output is a decision-support review queue with reason codes, actions and confidence labels, not a causal claim about refreshes.

## 1. Problem framing

- **Decision:** which 10-20 pages, out of hundreds per client, an editor reviews this month, and with which action (refresh, expand, monitor, protect).
- **Unit:** one page at a monthly decision moment, ranked inside its client.
- **Output:** a ranked queue per client with reason codes, an action and a confidence label.
- **Cost of a wrong call:** a false positive wastes editor hours (and editing a healthy page can hurt it); a false negative lets a slide continue unnoticed. Lost visibility is concentrated in few pages, so the top of the list matters most, which is what precision@20 measures.
- **Why data helps:** simple rules people reach for ("biggest first", "oldest first") point at the wrong pages (ML-03); the in-month slide signal had to be found and validated in the data.

## 2. Data safety

- **Used:** `fact_content_daily_performance` (Feb-Jun 2026 partitions) and `dim_content` (creation date, keyword metadata).
- **Excluded:**
  - all label-month columns
  - `content_updated_date` and `last_optimized_date` (snapshot dates postdating the decision for 78% of pages)
  - GA4 columns (about 4% available)
  - IDs as features
- **Leakage risks checked:**
  - deliberate leak → ROC-AUC 0.999, removed
  - random vs grouped split gap (0.72 vs 0.63 ROC-AUC)
  - keyword metadata ablation (under 1 point)
  - population filter using label-month coverage, disclosed (drops 18.5%)
- **Nothing client-identifying** appears in `work/` or `docs/index.html`. IDs are pseudonymous hashes and are never printed in reviews. Data files are gitignored, and only aggregate JSON receipts are committed.

## 3. Baseline

ML-07 rule (`work/notebooks/w04_baseline_score.ipynb`): score = impressions per day lost between the first and second half of the month, ×1.5 when CTR is below its position-tier median. It is built on two signal checks, momentum (CONFIRMED) and CTR-vs-position (CONFIRMED).

On March → April, evaluated over 22 clients with 20+ pages:

| Ranking | Precision@20 | Loss captured |
|---|---|---|
| Base rate | 0.489 | n/a |
| ML-07 rule | 0.711 | 0.241 |
| Random order | n/a | 0.104 |
| Biggest pages first | 0.427 | n/a |

## 4. Model / analysis

- **Model:** logistic regression on 16 features: impressions, clicks, CTR, impression-weighted position, momentum, within-month drop, days with impressions, daily volatility, position change, biggest-day share, page age, search volume (and whether it is present), competition, keyword length, content type, intent.
- **Ranking score:** probability × (1 + within-month drop per day).
- **Also tried:** gradient-boosted classifier and expected-loss regressor.
- **Target:** next-month average daily impressions below 80% of this month's.

## 5. Evaluation

- **Split design:**
  - GroupKFold(5) by client on March → April
  - time-aware (train Feb → Mar, test Mar → Apr)
  - sealed May → Jun, run once
  - client bootstrap (2,000 resamples)
- **Precision@20, rule vs model:**
  - dev 0.711 vs 0.718
  - time-aware 0.711 vs 0.720
  - sealed 0.720 vs 0.732
- **Model − rule, 95% intervals:** precision includes zero in both periods. Loss captured is tied in dev; the rule is better on the sealed month (−0.016 [−0.029, −0.004]).
- **Rule − base rate:** +0.22 to +0.24, clearly above zero.
- **Errors:**
  - the most common miss is a spike early in the month that fades and recovers
  - clients under ~100 pages see no lift
  - probabilities do not transfer between months (decline rate 21% → 47%), so only ranks are used

## 6. Interpretation

- **The slide is the signal:** the size of the slide already visible within the month carries most of the ranking signal.
- **Logistic weights:** a big drop, high impressions and a single-day spike lean toward decline; clicks and momentum lean toward holding. High impressions with few clicks acts like a CTR warning.
- **Negative result:** the learned models did not measurably beat a transparent rule.

## 7. Recommendation

- **Start with the top 10-20 per client, high confidence first.** In April, 79% of high-confidence picks declined, against 60% for low.
- **Actions:**
  - refresh sliding page-one pages
  - expand sliding positions 11-20
  - monitor spikes and deep positions
  - protect steady page-one pages
- **Before acting:** check daily impressions in Search Console.
- **Where not to use it:** skip clients under ~100 pages, and never show the score as a probability or automate edits.
- **Monthly check:** if precision@20 falls below the rule for two months, ship the rule.

## 8. Reproducibility

```bash
git clone https://github.com/sohamrathi17/flyrank-ml-internship && cd flyrank-ml-internship
pip install -r requirements.txt duckdb jupyter
hf auth login            # READ token, after accepting the dataset gate
# run in order: work/notebooks/w03_data_contract, w04_baseline_score, w05_model,
# w06_validation_audit, w07_action_playbook, capstone (rebuilds docs/index.html)
```

- **Seed:** 42 everywhere. Python 3.9, scikit-learn 1.6.1, pandas 2.3, duckdb 1.4.
- **Data:** month files download once into the Hugging Face cache.
- **Sealed test:** the frame-builder is `work/scripts/sealed_test.py`, and its receipt is `work/outputs/sealed_test_metrics.json`. The capstone notebook re-computes it and asserts it matches.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).
