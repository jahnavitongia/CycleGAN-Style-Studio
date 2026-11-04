"""Compare routers on the 30-request test set.
Metrics: overall accuracy, in-scope accuracy, HALLUCINATION RATE (out-of-scope request routed to a model
instead of 'none'), median latency.   Usage:  python eval_router.py   (needs LLM_API_KEY for LLM rows)
Writes results.md and failures.json for the PDF report."""
import json
import os
import statistics
import time

import routers
from data import TEST
from registry import LABELS

REAL = set(LABELS) - {"none"}

SLEEP = float(os.getenv("EVAL_SLEEP", "0.5"))   # be gentle with free-tier rate limits


def available(method):
    if method in ("zero_shot", "few_shot"):
        return bool(os.getenv("LLM_API_KEY"))
    if method == "bert":
        return routers.bert_available()
    return True


rows, failures = [], []
for method, fn in routers.ROUTERS.items():
    if not available(method):
        print(f"skip {method} (not available)")
        continue
    preds, lat = [], []
    for req, gold in TEST:
        t0 = time.perf_counter()
        try:
            pred = fn(req)["model"]
        except Exception as e:
            pred = f"ERROR({type(e).__name__})"
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
        if pred != gold:
            failures.append({"method": method, "request": req, "expected": gold, "got": pred})
        if method in ("zero_shot", "few_shot"):
            time.sleep(SLEEP)
    ins = [(p, g) for p, (_, g) in zip(preds, TEST) if g != "none"]
    oos = [(p, g) for p, (_, g) in zip(preds, TEST) if g == "none"]
    rows.append({
        "method": method,
        "accuracy": sum(p == g for p, (_, g) in zip(preds, TEST)) / len(TEST),
        "in_scope_acc": sum(p == g for p, g in ins) / len(ins),
        "hallucination_rate": sum(p in REAL for p, _ in oos) / len(oos),
        "median_ms": statistics.median(lat),
    })

md = ["| Router | Accuracy (30) | In-scope acc (20) | Hallucination rate (10 out-of-scope) | Median latency |",
      "|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['method']} | {r['accuracy']:.0%} | {r['in_scope_acc']:.0%} | "
              f"{r['hallucination_rate']:.0%} | {r['median_ms']:.0f} ms |")
open("results.md", "w").write("\n".join(md) + "\n")
json.dump(failures, open("failures.json", "w"), indent=2)
print("\n".join(md))
print(f"\n{len(failures)} failures -> failures.json (pick ONE for the report's failure-case section)")
