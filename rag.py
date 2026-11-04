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

    def build(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])
        return self

    def search(self, q, k=6):
        if self.mat is None:
            self.build()
        sims = linear_kernel(self.vec.transform([q]), self.mat).ravel()
        idx = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in idx if sims[i] > 0]

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
                "bad_citations": [int(n) for n in re.findall(r"\[(\d+)\]", ans) if int(n) > len(hits)]}

    @staticmethod
    def check_numbers(answer, hits):
        """Numbers in the answer that appear in none of the retrieved passages (possible hallucination)."""
        pool = " ".join(h["text"] for h in hits).replace(",", "")
        nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", re.sub(r"\[\d+\]", "", answer))}
        return sorted(n for n in nums if len(n) > 1 and n.rstrip(".") not in pool)

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
