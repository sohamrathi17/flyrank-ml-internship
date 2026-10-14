# Which Pages Should You Refresh First?

**FlyRank ML Internship: Applied Search Intelligence · Soham Rathi**
Lane: **Refresh / Content Opportunity Scoring**

📄 **Paper:** [sohamrathi17.github.io/flyrank-ml-internship](https://sohamrathi17.github.io/flyrank-ml-internship/)
📝 **Capstone report:** [`work/capstone_report.md`](work/capstone_report.md)
📓 **All my work:** [`work/`](work/README.md)

---

## In short

Content editors can review only a handful of a site's pages each month. I built and validated a **monthly review queue** that ranks each client's pages so the ones most likely to keep losing search visibility next month come first.

- **Data:** the FlyRank ML Internship warehouse, about 79 million rows of pseudonymised daily Google Search Console data. Each snapshot takes features from one month and the outcome from the next.
- **Method:** a transparent baseline rule against logistic regression and gradient-boosted trees on 16 signals. Validation uses:
  - client-grouped cross-validation
  - a time-aware split
  - a **sealed final month evaluated once**
  - a client-level bootstrap
- **Result:** ranking pages by how much they had already slid within the month put **71-72% true decliners** in each client's top 20, against a **~49% base rate**, and held up on the sealed month. The learned models were not measurably better than this rule, and the paper reports that plainly.
- **Output:** a decision-support queue with actions (refresh / expand / monitor / protect), reason codes and confidence labels, plus a human-review checklist and monthly monitoring triggers.

| Precision@20 per client | Rule | Model | Base rate |
|---|---|---|---|
| Development month (Mar → Apr), grouped CV | 0.711 | 0.718 | 0.489 |
| Time-aware (trained Feb → Mar) | 0.711 | 0.720 | 0.489 |
| **Sealed month (May → Jun), run once** | **0.720** | **0.732** | 0.479 |

All claims are observational and decision-support only: no causal claims about refreshes, and nothing about Google's ranking system.

## Where to look

| What | Where |
|---|---|
| The paper (all 9 sections, interactive charts) | [GitHub Pages](https://sohamrathi17.github.io/flyrank-ml-internship/) · source built from [`work/scripts/paper_template.html`](work/scripts/paper_template.html) |
| Assignment notebooks ML-02 to ML-12, with outputs | [`work/notebooks/`](work/notebooks/), indexed in [`work/README.md`](work/README.md) |
| Shared code: slice builder, baseline rule, models, metrics | [`work/scripts/lane_data.py`](work/scripts/lane_data.py) |
| Sealed final-month test | [`work/scripts/sealed_test.py`](work/scripts/sealed_test.py) → [`work/outputs/sealed_test_metrics.json`](work/outputs/sealed_test_metrics.json) |
| Metric receipts behind every number in the paper | [`work/outputs/*.json`](work/outputs/) |
| Week 1-2 course notebooks (with my experiments) | [`notebooks/01`](notebooks/01_first_look_and_discovery.ipynb), [`notebooks/02`](notebooks/02_your_first_readable_model.ipynb) |
| Paper URL for submission | [`submission/paper_url.txt`](submission/paper_url.txt) |

## Reproduce

```bash
git clone https://github.com/sohamrathi17/flyrank-ml-internship && cd flyrank-ml-internship
pip install -r requirements.txt duckdb jupyter
hf auth login     # a READ token, after accepting the dataset gate on Hugging Face
```

Then run the notebooks in `work/notebooks/` in order (`w03` → `w07`, then `capstone`). Month files download once into the Hugging Face cache, outside the repo. The capstone notebook re-checks the sealed result against its receipt and rebuilds `docs/index.html`. Seed 42 throughout; Python 3.9, scikit-learn 1.6.1.

Data files never enter git: `.gitignore` and the CI leak guard block them. Only small aggregate JSON receipts are committed.

## About this repository

This repo started from FlyRank's [internship starter template](https://github.com/flyrank-bih/flyrank-ml-internship-starter). The parts that came with it are kept unchanged as reference:
- the starter pipeline in `scripts/`
- the anonymised starter CSV in `data/`
- the course guides: [`SETUP.md`](SETUP.md), [`GUIDE.md`](GUIDE.md), [`DATA_USE.md`](DATA_USE.md) and `docs/`
- the skill notes in `skills/`

My work is everything in `work/`, the paper in `docs/index.html`, and the experiments in notebooks 01-02.

## Data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). All client and page identifiers are pseudonymous; no client names, domains, URLs or search queries appear in this work.
