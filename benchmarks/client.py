"""Minimal client for a Jev-shaped `/v1/systemone` endpoint, plus the metrics every runner reports.

Answer normalisation: every answer becomes a probability dict over the question's own option
keys (noul -> {"true": p, "false": 1-p}; choice -> probabilities as returned; score ->
probabilities keyed by level index as strings), so the scorers below work for all three types.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class DecisionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8008", model: str = "laya", timeout: float = 300.0,
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def systemone(self, state: Any, questions: Dict[str, dict]) -> Tuple[dict, float]:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:400]!r}") from None
        return out, time.perf_counter() - t0


def answer_probs(q: dict, answer: dict) -> Dict[str, float]:
    """Probabilities over the question's option keys, in a type-independent dict."""
    t = q["type"]
    if t == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities") or {}
    if t == "score":
        return {str(k): float(v) for k, v in probs.items()}
    return {str(k): float(v) for k, v in probs.items()}


def question_keys(q: dict) -> List[str]:
    t = q["type"]
    if t == "noul":
        return ["false", "true"]
    crit = q.get("criteria")
    if t == "score":
        return [str(i) for i in range(len(crit))]
    return [str(k) for k in (crit.keys() if isinstance(crit, dict) else crit)]


# ----------------------------------------------------------------------------------- metrics
def nll(probs: Dict[str, float], gold_key: str) -> float:
    return -math.log(max(probs.get(gold_key, 0.0), 1e-9))


def brier_hard(probs: Dict[str, float], gold_key: str) -> float:
    """Multi-class Brier against a one-hot gold, summed over options (0 = perfect, 2 = worst)."""
    return sum((p - (1.0 if k == gold_key else 0.0)) ** 2 for k, p in probs.items())


def brier_soft(probs: Dict[str, float], gold: Dict[str, float]) -> float:
    keys = set(probs) | set(gold)
    return sum((probs.get(k, 0.0) - gold.get(k, 0.0)) ** 2 for k in keys)


def kl_to_gold(probs: Dict[str, float], gold: Dict[str, float]) -> float:
    """KL(gold || model), the typed-decisions leaderboard's 'KL from gold'."""
    return sum(g * math.log(max(g, 1e-9) / max(probs.get(k, 0.0), 1e-9)) for k, g in gold.items() if g > 0)


def ece(conf_correct: List[Tuple[float, bool]], n_bins: int = 10) -> Optional[float]:
    if not conf_correct:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, ok in conf_correct:
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    total = len(conf_correct)
    return sum(len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
               for b in bins if b)


def argmax(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)


def summarize(records: List[dict]) -> dict:
    """records: dicts with probs, gold (key), optional gold_dist, latency_s, type."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    acc = sum(r["pred"] == r["gold"] for r in records) / n
    out = {
        "n": n,
        "accuracy": round(acc, 4),
        "nll": round(sum(r["nll"] for r in records) / n, 4),
        "brier": round(sum(r["brier"] for r in records) / n, 4),
        "ece": round(ece([(r["conf"], r["pred"] == r["gold"]) for r in records]), 4),
        "mean_confidence": round(sum(r["conf"] for r in records) / n, 4),
    }
    if all("kl" in r for r in records):
        out["kl_from_gold"] = round(sum(r["kl"] for r in records) / n, 4)
        out["brier_soft"] = round(sum(r["brier_soft"] for r in records) / n, 4)
    lat = sorted(r["latency_s"] for r in records if "latency_s" in r)
    if lat:
        out["latency_ms_p50"] = round(1000 * lat[len(lat) // 2], 1)
        out["latency_ms_p90"] = round(1000 * lat[int(len(lat) * 0.9)], 1)
    return out


def record(q: dict, answer: dict, gold_key: str, gold_dist: Optional[Dict[str, float]] = None, **extra) -> dict:
    probs = answer_probs(q, answer)
    pred = argmax(probs)
    r = {"probs": probs, "pred": pred, "gold": str(gold_key), "conf": probs[pred],
         "nll": nll(probs, str(gold_key)), "brier": brier_hard(probs, str(gold_key)), "type": q["type"]}
    if gold_dist is not None:
        gd = {str(k): float(v) for k, v in gold_dist.items()}
        r["kl"] = kl_to_gold(probs, gd)
        r["brier_soft"] = brier_soft(probs, gd)
    r.update(extra)
    return r
