"""Retrieval-augmented research over filings/news/transcripts. TF-IDF retrieval (no GPU, no API),
every chunk carries a citation; answers are checked so cited numbers must appear in the sources."""
import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from edgar import SECTIONS, extract_sections


def _chunks(text, size=900):
    paras, cur, out = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()], "", []
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur); cur = cur[-150:] + " " + p   # small overlap
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        out.append(cur)
    return out


class FilingIndex:
    def __init__(self):
        self.chunks, self.vec, self.mat = [], None, None

    def add_filing(self, meta, text):
        """meta: {'form','date','url'}; splits into 10-K items when possible."""
        secs = extract_sections(text) if meta.get("form") in ("10-K", "10-Q") else {}
        for item, body in (secs or {"": text}).items():
            for c in _chunks(body):
                self.chunks.append({"text": c, "form": meta.get("form"), "date": meta.get("date"),
                                    "item": item, "section": SECTIONS.get(item, ""), "url": meta.get("url"),
                                    "kind": "filing"})

    def add_document(self, title, text, date="", url="", kind="news"):
        for c in _chunks(text):
            self.chunks.append({"text": c, "form": title, "date": date, "item": "", "section": "", "url": url, "kind": kind})

    RELIABILITY = {"statement": 1.0, "filing": 1.0, "release": 0.95, "transcript": 0.9, "news": 0.6}   # prior trust by source kind (heuristic, adjustable)

    def add_statements(self, periods, ticker=""):
        """Table-aware retrieval: every financial-statement line becomes its own short chunk ('Revenue FY2024 ... = 60.9B; change vs prior ...'),
        plus derived margins, so numeric questions retrieve exact rows instead of prose. periods: fin_model.normalize() output, newest first."""
        lines = {"revenue": "Revenue", "gross_profit": "Gross profit", "ebit": "Operating income", "net_income": "Net income",
                 "ocf": "Operating cash flow", "total_assets": "Total assets", "total_liabilities": "Total liabilities", "cash": "Cash", "sga": "SG&A", "cogs": "Cost of revenue"}
        for i, p in enumerate(periods):
            prev = periods[i + 1] if i + 1 < len(periods) else None
            yr = p["date"][:4]
            rows = []
            for k, label in lines.items():
                if p.get(k) is None:
                    continue
                s = f"{ticker} {label} FY{yr} ({p['date']}) was {p[k]:,.0f}."
                if prev and prev.get(k):
                    s += f" Change vs FY{prev['date'][:4]}: {100 * (p[k] - prev[k]) / abs(prev[k]):+.1f}%."
                rows.append(s)
            if p.get("revenue"):
                for k, label in (("gross_profit", "Gross margin"), ("ebit", "Operating margin"), ("net_income", "Net margin")):
                    if p.get(k) is not None:
                        s = f"{ticker} {label} FY{yr} was {100 * p[k] / p['revenue']:.1f}%."
                        if prev and prev.get("revenue") and prev.get(k) is not None:
                            s += f" Prior year {100 * prev[k] / prev['revenue']:.1f}% (change {100 * (p[k] / p['revenue'] - prev[k] / prev['revenue']):+.1f} pp)."
                        rows.append(s)
            for s in rows:
                self.chunks.append({"text": s, "form": "Financial statements", "date": p["date"], "item": "", "section": "table", "url": "", "kind": "statement"})

    def add_transcript(self, title, text, date="", url=""):
        """Earnings-call transcript text supplied by the user (not fetched automatically)."""
        for c in _chunks(text, 700):
            self.chunks.append({"text": c, "form": title, "date": date, "item": "", "section": "", "url": url, "kind": "transcript"})

    def build(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])
        self._tok = [re.findall(r"\w+", c["text"].lower()) for c in self.chunks]
        self._emb, self._embed_kind = self._build_embeddings()
        self._df, self._avgdl = {}, (np.mean([len(t) for t in self._tok]) if self._tok else 1)
        for t in self._tok:
            for w in set(t):
                self._df[w] = self._df.get(w, 0) + 1
        return self

    def _build_embeddings(self):
        """Dense vectors. sentence-transformers (all-MiniLM-L6-v2) when installed; otherwise LSA (TruncatedSVD of TF-IDF), which is offline and
        captures co-occurrence semantics but is weaker than a neural encoder."""
        texts = [c["text"] for c in self.chunks]
        if not texts:
            return None, None
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            return np.asarray(self._st.encode(texts, normalize_embeddings=True, show_progress_bar=False)), "sentence-transformers"
        except Exception:
            pass
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
            k = min(128, max(2, min(self.mat.shape) - 1))
            self._svd = TruncatedSVD(k, random_state=0).fit(self.mat)
            return normalize(self._svd.transform(self.mat)), "lsa"
        except Exception:
            return None, None

    def _embed_query(self, q):
        if self._embed_kind == "sentence-transformers":
            return np.asarray(self._st.encode([q], normalize_embeddings=True, show_progress_bar=False))[0]
        from sklearn.preprocessing import normalize
        return normalize(self._svd.transform(self.vec.transform([q])))[0]

    def _bm25(self, q, k1=1.5, b=0.75):
        n, qs = len(self._tok), set(re.findall(r"\w+", q.lower()))
        out = np.zeros(n)
        for i, t in enumerate(self._tok):
            if not t:
                continue
            tf = {}
            for w in t:
                if w in qs:
                    tf[w] = tf.get(w, 0) + 1
            for w, f in tf.items():
                idf = np.log(1 + (n - self._df[w] + .5) / (self._df[w] + .5))
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / self._avgdl))
        return out

    def search(self, q, k=6, rerank=None):
        """Hybrid retrieval: BM25 + TF-IDF + embeddings fused by reciprocal rank fusion, scaled by source reliability.
        rerank: optional callable(question, [texts]) -> scores (e.g. a cross-encoder); embeddings/cross-encoders plug in here."""
        if self.mat is None or not hasattr(self, "_tok"):
            self.build()
        tf = linear_kernel(self.vec.transform([q]), self.mat).ravel()
        bm = self._bm25(q)
        rankers = [tf, bm]
        if getattr(self, "_emb", None) is not None:
            rankers.append(self._emb @ self._embed_query(q))
        fused = np.zeros(len(tf))
        for s in rankers:
            order = np.argsort(-s)
            for rank, i in enumerate(order[:50]):
                if s[i] > 0:
                    fused[i] += 1.0 / (60 + rank)
        fused *= np.array([self.RELIABILITY.get(c.get("kind"), 0.7) for c in self.chunks])
        idx = [i for i in np.argsort(-fused)[:max(k * 3, k)] if fused[i] > 0]
        if rerank and idx:
            sc = rerank(q, [self.chunks[i]["text"] for i in idx])
            idx = [idx[j] for j in np.argsort(-np.asarray(sc))]
        return [{**self.chunks[i], "score": float(fused[i] * 1000)} for i in idx[:k]]

    def search_multi(self, q, per_kind=2, kinds=("statement", "filing", "transcript", "news")):
        """Multi-document retrieval: top hits from EACH source kind so an answer can combine statements, 10-K text, calls and news."""
        pool = self.search(q, k=len(self.chunks))
        out = []
        for kd in kinds:
            out += [h for h in pool if h.get("kind") == kd][:per_kind]
        return out

    def explain(self, question, llm=None, per_kind=2):
        """'Why did X change and what did management say?' -> sourced answer separating REPORTED FACTS (statements/filings/transcripts) from INTERPRETATION.
        Without an LLM, returns the evidence grouped by kind (still sourced)."""
        hits = self.search_multi(question, per_kind)
        if not hits:
            return {"answer": "No relevant passages.", "sources": []}
        src = "\n\n".join(f"[{i + 1}] ({h['kind']}: {self.cite(h)})\n{h['text']}" for i, h in enumerate(hits))
        if llm is None:
            groups = {}
            for i, h in enumerate(hits):
                groups.setdefault(h["kind"], []).append(f"{h['text'][:300]} [{i + 1}]")
            ans = "\n".join(f"{k.upper()}: " + " | ".join(v) for k, v in groups.items())
        else:
            ans = llm("Using ONLY the numbered sources, answer the question in two labelled parts. "
                      "REPORTED FACTS: numbers and statements quoted from statements/filings/transcripts, each with [n]. "
                      "INTERPRETATION: your reasoning linking them, clearly marked as interpretation, noting any gaps or conflicts.\n\n"
                      f"SOURCES:\n{src}\n\nQUESTION: {question}\nANSWER:")
        return {"answer": ans, "sources": [{"n": i + 1, "kind": h["kind"], "citation": self.cite(h), "excerpt": h["text"][:250]} for i, h in enumerate(hits)],
                "claim_checks": self.verify_claims(ans, hits), "possible_conflicts": self.detect_contradictions(hits)}

    @staticmethod
    def cite(h):
        sec = f", Item {h['item']} ({h['section']})" if h.get("item") else ""
        return f"{h['form']} {h['date']}{sec} {h['url'] or ''}".strip()

    def answer(self, question, llm=None, k=6):
        hits = self.search(question, k)
        if not hits:
            return {"answer": "No relevant passage found in the indexed documents.", "sources": [], "unsupported_numbers": []}
        src = "\n\n".join(f"[{i + 1}] ({self.cite(h)})\n{h['text']}" for i, h in enumerate(hits))
        if llm is None:      # extractive fallback: best sentences, each with its citation
            qw = set(re.findall(r"\w+", question.lower()))
            sents = []
            for i, h in enumerate(hits):
                for s in re.split(r"(?<=[.!?])\s+", h["text"]):
                    sc = len(qw & set(re.findall(r"\w+", s.lower()))) / (len(s.split()) ** .5 + 1)
                    sents.append((sc, f"{s.strip()} [{i + 1}]"))
            ans = " ".join(s for _, s in sorted(sents, reverse=True)[:4])
        else:
            prompt = ("Answer the question using ONLY the numbered sources. Cite sources like [1] after each claim. "
                      "If sources conflict, say so and cite both. If the sources do not answer it, say that.\n\n"
                      f"SOURCES:\n{src}\n\nQUESTION: {question}\nANSWER:")
            ans = llm(prompt)
        return {"answer": ans, "sources": [{"n": i + 1, "citation": self.cite(h), "score": h["score"],
                                            "excerpt": h["text"][:300]} for i, h in enumerate(hits)],
                "unsupported_numbers": self.check_numbers(ans, hits),
                "claim_checks": self.verify_claims(ans, hits), "possible_conflicts": self.detect_contradictions(hits),
                "bad_citations": [int(n) for n in re.findall(r"\[(\d+)\]", ans) if int(n) > len(hits)]}

    _SCALE = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6,
              "thousand": 1e3, "k": 1e3, "%": None, "percent": None}

    @classmethod
    def _numbers(cls, text):
        """-> [(value, unit)] where unit is 'pct' or 'abs' (absolute amounts scaled to units)."""
        out = []
        for m in re.finditer(r"\$?\(?(\d[\d,]*\.?\d*)\)?\s*(trillion|billion|million|thousand|bn|mn|%|percent|[tbmk]\b)?", text, re.I):
            raw = m.group(1).replace(",", "").rstrip(".")
            if not raw:
                continue
            v, u = float(raw), (m.group(2) or "").lower()
            if u in ("%", "percent"):
                out.append((v, "pct"))
            elif u in cls._SCALE:
                out.append((v * cls._SCALE[u], "abs"))
            else:
                out.append((v, "abs"))
        return out

    @classmethod
    def verify_claims(cls, answer, hits, tol=0.02):
        """Per-sentence claim check. For every sentence with numbers a claim passes only if
        (1) the cited passage (or any passage if uncited) contains a number with the SAME VALUE after unit scaling and the SAME KIND (% vs amount),
        (2) that number sits in a sentence sharing >=2 content words with the claim (context/meaning),
        (3) any 4-digit fiscal year in the claim also appears in that passage (period).
        Entity names are checked when the answer names a capitalised entity absent from every passage.
        Returns [{claim, status: supported|unsupported|weak, reason}]. Heuristic: it catches mismatched numbers/periods/units;
        it cannot prove semantic truth, so 'supported' means 'consistent with the cited text'."""
        stop = set("the a an of in on for to and was were is are by at as with from that this its it be has have had than".split())
        words = lambda s: {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in stop}
        res, pool_all = [], " ".join(h["text"] for h in hits)
        for sent in re.split(r"(?<=[.!?])\s+", answer):
            nums = [n for n in cls._numbers(re.sub(r"\[\d+\]", "", sent)) if not (1900 <= n[0] <= 2100 and n[1] == "abs" and float(n[0]).is_integer())]
            nums = [n for n in nums if n[0] >= 1 or n[1] == "pct"]
            if not nums:
                continue
            cites = [int(c) for c in re.findall(r"\[(\d+)\]", sent)]
            cand = [hits[c - 1] for c in cites if 0 < c <= len(hits)] or list(hits)
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", sent))
            problems = []
            for val, kind in nums:
                found = False
                for h in cand:
                    for ps in re.split(r"(?<=[.!?])\s+", h["text"]):
                        if any(k == kind and abs(v - val) <= tol * max(abs(val), 1e-9) for v, k in cls._numbers(ps)) \
                           and len(words(ps) & words(sent)) >= 2:
                            found = True
                            if years and not years <= set(re.findall(r"\b(?:19|20)\d{2}\b", h["text"])):
                                problems.append(f"period {sorted(years)} not found in cited passage")
                            break
                    if found:
                        break
                if not found:
                    problems.append(f"{val:g} ({'percent' if kind == 'pct' else 'amount'}) not found with matching context")
            ents = {e for e in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", re.sub(r"\[\d+\]", "", sent))[1:]}
            missing = [e for e in ents if e.lower() not in pool_all.lower()]
            if missing and len(missing) >= 2:
                problems.append(f"entities not in sources: {missing[:3]}")
            status = "supported" if not problems else ("unsupported" if any("not found with" in p for p in problems) else "weak")
            res.append({"claim": sent.strip(), "status": status, "reason": "; ".join(problems) or "numbers, units, period and context match cited text"})
        return res

    @staticmethod
    def detect_contradictions(hits):
        """Flag passages that report different values for the same metric phrase in different periods/sources (candidate conflicts for a human to read)."""
        pat = re.compile(r"((?:total |net |operating |gross )?(?:revenue|net income|operating income|operating margin|gross margin|free cash flow)[^.]{0,40}?)(\d[\d,]*\.?\d*)\s*(%|billion|million)?", re.I)
        seen, out = {}, []
        for i, h in enumerate(hits):
            for m in pat.finditer(h["text"]):
                key = re.sub(r"\s+", " ", m.group(1).lower().strip())[:30]
                val = (float(m.group(2).replace(",", "")), (m.group(3) or "").lower())
                if key in seen and seen[key][1][1] == val[1] and abs(seen[key][1][0] - val[0]) > 0.02 * max(val[0], 1):
                    out.append({"metric": key, "a": {"source": seen[key][0] + 1, "value": seen[key][1]}, "b": {"source": i + 1, "value": val},
                                "note": "differing values; may be different periods, restatement, or a conflict"})
                seen.setdefault(key, (i, val))
        return out

    @classmethod
    def check_numbers(cls, answer, hits):
        """Back-compat: numbers in claims that failed claim-level verification."""
        bad = [c for c in cls.verify_claims(answer, hits) if c["status"] != "supported"]
        return [c["reason"] for c in bad]

    def save(self, path):
        pickle.dump(self.chunks, open(path, "wb"))

    @classmethod
    def load(cls, path):
        i = cls(); i.chunks = pickle.load(open(path, "rb")); return i.build()


def make_llm(spec):
    """'ollama:qwen2.5:7b' or a Gemini model name; None -> extractive answers."""
    if not spec:
        return None
    if spec.startswith("ollama:"):
        import requests
        name = spec.split(":", 1)[1]
        return lambda p: requests.post("http://localhost:11434/api/generate", timeout=600,
                                       json={"model": name, "prompt": p, "stream": False,
                                             "options": {"temperature": 0}}).json()["response"]
    from utils import client
    return lambda p: client.models.generate_content(model=spec, contents=p).text


def make_reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Optional cross-encoder reranker (pip install sentence-transformers). Returns callable for FilingIndex.search(rerank=...); None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model)
        return lambda q, texts: ce.predict([(q, t) for t in texts])
    except Exception:
        return None
