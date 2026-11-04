from finclaim.localllm import make_gen
"""python -m finclaim.run build --tickers AAPL MSFT ... [--synthetic]
   python -m finclaim.run eval  --data data/finclaim.jsonl --systems tool judge closedbook [--split test] [--limit 200]
   python -m finclaim.run labelsheet --data data/finclaim.jsonl --n 100        (blind sheet for a human labeller)
   python -m finclaim.run agreement --data data/finclaim.jsonl --sheet labelsheet_filled.csv"""
import argparse
import csv
import os
import random
from . import claims, facts, metrics
from .verifiers import ToolVerifier, LLMJudge, Majority, gemini_generate


def cmd_build(a):
    recs = facts.synthetic(12) if a.synthetic else []
    for t in ([] if a.synthetic else a.tickers):
        try:
            r = facts.load_yf(t)
            if len(r["facts"]) >= 6:
                recs.append(r)
            else:
                print("skipped (too little data):", t)
        except Exception as e:
            print("skipped", t, e)
    rows = claims.build(recs, a.n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    claims.save(rows, a.out)
    print(f"{len(recs)} companies, {len(rows)} claims -> {a.out}")


def cmd_eval(a):
    rows = [r for r in claims.load(a.data) if a.split == "all" or r["split"] == a.split]
    if a.limit:
        random.Random(0).shuffle(rows)
        rows = rows[:a.limit]
    sysmap = {"majority": lambda: Majority(), "tool": lambda: ToolVerifier(),
              "judge": lambda: LLMJudge(make_gen(a.model), f"gemini_with_evidence_{a.model}", True, "judge_cache.jsonl", a.pause),
              "closedbook": lambda: LLMJudge(make_gen(a.model), f"gemini_closed_book_{a.model}", False, "judge_cache.jsonl", a.pause)}
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(rows)} claims, {len({r['ticker'] for r in rows})} companies, split={a.split}")
    for s in a.systems:
        sysm = sysmap[s]()
        preds = [sysm.predict(r) for r in rows]
        y = [r["label"] for r in rows]
        lo, hi = metrics.cluster_bootstrap(rows, preds)
        print(f"\n== {sysm.name}: accuracy {metrics.accuracy(y, preds):.3f} (90% CI {lo:.3f}-{hi:.3f}), macro-F1 {metrics.macro_f1(y, preds):.3f}")
        print("   REFUTED recall by error type:", {k: round(v, 2) for k, v in metrics.recall_by_error(rows, preds).items()})
        with open(f"{a.out}/preds_{sysm.name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "ticker", "label", "pred", "error_type", "claim"])
            for r, p in zip(rows, preds):
                w.writerow([r["id"], r["ticker"], r["label"], p, r["error_type"], r["claim"]])


def cmd_labelsheet(a):
    rows = claims.load(a.data)
    random.Random(1).shuffle(rows)
    with open("labelsheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "evidence", "claim", "your_label (SUPPORTED/REFUTED/NOT_ENOUGH_INFO)"])
        for r in rows[:a.n]:
            w.writerow([r["id"], r["evidence"], r["claim"], ""])
    print("wrote labelsheet.csv: fill the last column from the evidence only, without looking at any model output")


def cmd_agreement(a):
    gold = {r["id"]: r["label"] for r in claims.load(a.data)}
    ids, human = [], []
    for row in csv.reader(open(a.sheet)):
        if row[0] == "id" or not row[3].strip():
            continue
        ids.append(int(row[0])), human.append(row[3].strip().upper())
    g = [gold[i] for i in ids]
    print(f"{len(ids)} labelled; agreement {metrics.accuracy(g, human):.3f}; Cohen's kappa {metrics.cohen_kappa(g, human):.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--tickers", nargs="*", default=[]); b.add_argument("--n", type=int, default=30)
    b.add_argument("--out", default="data/finclaim.jsonl"); b.add_argument("--synthetic", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("--data", default="data/finclaim.jsonl")
    e.add_argument("--systems", nargs="+", default=["majority", "tool"]); e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=0); e.add_argument("--model", default="gemini-2.5-flash"); e.add_argument("--pause", type=float, default=7.0); e.add_argument("--out", default="results")
    l = sub.add_parser("labelsheet"); l.add_argument("--data", default="data/finclaim.jsonl"); l.add_argument("--n", type=int, default=100)
    g = sub.add_parser("agreement"); g.add_argument("--data", default="data/finclaim.jsonl"); g.add_argument("--sheet", required=True)
    a = p.parse_args()
    {"build": cmd_build, "eval": cmd_eval, "labelsheet": cmd_labelsheet, "agreement": cmd_agreement}[a.cmd](a)


if __name__ == "__main__":
    main()
