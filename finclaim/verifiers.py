"""Systems under test. Each takes a claim record (with `claim` and `evidence`) and returns a label."""
import json
import os
import re
import time
from .facts import ITEMS, MARGINS

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def parse_evidence(text):
    ev = {}
    for line in text.splitlines()[1:]:
        item, _, rest = line.partition(": ")
        ev[item] = {int(y): float(v.replace(",", "")) for y, v in re.findall(r"FY(\d{4})=(-?[\d,\.]+)", rest)}
    return ev


class ToolVerifier:
    """Deterministic 'tool use' baseline: parse the claim, recompute from the evidence table, compare.
    It only understands the templates in claims.py, so it is an upper bound on template claims, NOT a general verifier."""
    name = "tool_recompute"

    def predict(self, r):
        ev, c = parse_evidence(r["evidence"]), r["claim"]
        key = next((k for k in sorted(list(MARGINS) + list(ITEMS), key=len, reverse=True) if f"'s {k} " in c), None)
        years = [int(y) for y in re.findall(r"fiscal (\d{4})", c)]
        if key is None or not years:
            return "NOT_ENOUGH_INFO"
        y = years[-1]
        if key in MARGINS:
            a, b = MARGINS[key]
            if y not in ev.get(a, {}) or y not in ev.get(b, {}):
                return "NOT_ENOUGH_INFO"
            truth, num = 100 * ev[a][y] / ev[b][y], float(re.search(r"was (-?[\d\.]+)%", c).group(1))
            return "SUPPORTED" if abs(truth - num) <= 0.06 else "REFUTED"
        s = ev.get(key, {})
        if re.search(r"(grew|fell) ", c):
            if y not in s or y - 1 not in s:
                return "NOT_ENOUGH_INFO"
            truth = 100 * (s[y] / s[y - 1] - 1)
            m = re.search(r"(grew|fell) ([\d\.]+)%", c)
            claimed = float(m.group(2)) * (1 if m.group(1) == "grew" else -1)
            return "SUPPORTED" if abs(truth - claimed) <= 0.06 else "REFUTED"
        if y not in s:
            return "NOT_ENOUGH_INFO"
        m = re.search(r"was \$([\d,\.]+) (billion|million)", c)
        if not m:
            return "NOT_ENOUGH_INFO"
        claimed = float(m.group(1).replace(",", "")) * (1000 if m.group(2) == "billion" else 1)
        tol = 0.5 * (100 if m.group(2) == "billion" else 1) * 1.01
        return "SUPPORTED" if abs(s[y] - claimed) <= tol else "REFUTED"


PROMPT = """You are verifying a claim about a company's financials.
Use ONLY the evidence. Compute percentages and margins yourself when needed.
Answer with exactly one word: SUPPORTED (evidence confirms it, allowing for rounding),
REFUTED (evidence contradicts it), or NOT_ENOUGH_INFO (evidence does not contain what is needed).

EVIDENCE:
{evidence}

CLAIM: {claim}

ANSWER:"""


class LLMJudge:
    """LLM judge with disk cache. generate: callable(prompt)->text. with_evidence=False gives the closed-book baseline."""
    def __init__(self, generate, name="llm_judge", with_evidence=True, cache=None, pause=0.0):
        self.generate, self.name, self.with_evidence, self.pause = generate, name, with_evidence, pause
        self.cache_path, self.cache = cache, {}
        if cache and os.path.exists(cache):
            self.cache = {json.loads(l)["k"]: json.loads(l)["v"] for l in open(cache)}

    def predict(self, r):
        ev = r["evidence"] if self.with_evidence else "(no evidence provided)"
        prompt = PROMPT.format(evidence=ev, claim=r["claim"])
        k = f"{self.name}|{hash_text(prompt)}"
        if k not in self.cache:
            out = (self.generate(prompt) or "").upper()
            lab = next((l for l in ("NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "REFUTED", "SUPPORTED") if l in out), "NOT_ENOUGH_INFO")
            self.cache[k] = lab.replace(" ", "_")
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"k": k, "v": self.cache[k]}) + "\n")
            time.sleep(self.pause)
        return self.cache[k]


def hash_text(s):
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()


def gemini_generate(model="gemini-2.5-flash"):
    """Uses the client already configured in your project's utils.py (GEMINI_API_KEY from .env)."""
    from utils import client

    def g(prompt):
        for a in range(6):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except Exception as e:
                print(f"  Gemini error (attempt {a+1}/6): {str(e)[:300]}", flush=True)
                time.sleep(15 * (a + 1))
        raise RuntimeError("Gemini failed repeatedly; see the error above")
    return g


class Majority:
    name = "always_SUPPORTED"

    def predict(self, r):
        return "SUPPORTED"
