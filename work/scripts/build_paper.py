"""Builds docs/index.html (the deployed paper) from work/scripts/paper_template.html.

Every chart number on the page comes from the committed receipts in work/outputs/, injected here,
so the paper cannot drift from the notebooks. Run from the repo root (the capstone notebook calls it).
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "work", "outputs")
RECEIPTS = ["baseline_metrics", "model_metrics", "validation_metrics", "paper_data",
            "sealed_test_metrics", "capstone_metrics"]


def main():
    data = {name: json.load(open(os.path.join(OUT, f"{name}.json"))) for name in RECEIPTS}
    template = open(os.path.join(os.path.dirname(__file__), "paper_template.html"), encoding="utf-8").read()
    assert "__PAPER_DATA__" in template
    html = template.replace("__PAPER_DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"paper build: docs/index.html ({len(html) / 1024:.0f} KB) from {len(RECEIPTS)} receipts")


if __name__ == "__main__":
    main()
