"""python -m finclaim.tests   (offline, synthetic companies)"""
import numpy as np
from . import claims, facts, metrics
from .verifiers import ToolVerifier, LLMJudge, Majority, parse_evidence


def main():
    recs = facts.synthetic(14, seed=1)
    rows = claims.build(recs, 30, seed=2)
    labs = [r["label"] for r in rows]
    assert {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"} == set(labs)
    # splits by company, no overlap
    sp = {}
    for r in rows:
        sp.setdefault(r["ticker"], set()).add(r["split"])
    assert all(len(v) == 1 for v in sp.values())
    # evidence round-trips
    ev = parse_evidence(rows[0]["evidence"])
    assert set(ev) == set(recs[0]["facts"])
    # NEI claims mention items/years absent from the evidence
    for r in rows:
        if r["label"] == "NOT_ENOUGH_INFO":
            assert ToolVerifier().predict(r) == "NOT_ENOUGH_INFO", r["claim"]
    # the tool recomputation reproduces every gold label on template claims (checks label correctness)
    tv = ToolVerifier()
    preds = [tv.predict(r) for r in rows]
    bad = [(r["claim"], r["label"], p) for r, p in zip(rows, preds) if r["label"] != p]
    assert not bad, bad[:3]
    # sanity baselines and metrics
    assert metrics.accuracy(labs, [Majority().predict(r) for r in rows]) < 0.5
    assert abs(metrics.macro_f1(["SUPPORTED", "REFUTED"], ["SUPPORTED", "REFUTED"]) - 2 / 3) < 1e-9   # NEI class absent -> 0
    assert abs(metrics.cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) - 1.0) < 1e-9
    # a closed-book judge that always says SUPPORTED is caught by REFUTED recall = 0
    j = LLMJudge(lambda p: "SUPPORTED", "stub", with_evidence=False)
    pj = [j.predict(r) for r in rows]
    assert all(v == 0 for v in metrics.recall_by_error(rows, pj).values())
    lo, hi = metrics.cluster_bootstrap(rows, preds, n=50)
    assert lo <= 1.0 and hi <= 1.0
    print(f"ok | {len(rows)} claims, {len(recs)} companies | label mix",
          {l: labs.count(l) for l in set(labs)}, "| error types", sorted({r['error_type'] for r in rows}))


if __name__ == "__main__":
    main()
